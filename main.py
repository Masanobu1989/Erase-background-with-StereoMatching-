import cv2
import numpy as np
from PIL import Image, ExifTags
import math
from fractions import Fraction
import warnings

# ========== 便利関数 ==========
def read_exif(path):
    try:
        img = Image.open(path)
        exif = img._getexif() or {}
        # タグ名に変換
        exif_named = {}
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k, k)
            exif_named[tag] = v
        return exif_named
    except Exception as e:
        warnings.warn(f"EXIFを読めませんでした: {e}")
        return {}

def _as_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(t, (int, float)) for t in x):
        # (num, den) 形式
        return float(x[0]) / float(x[1] if x[1] != 0 else 1)
    try:
        return float(x)
    except:
        return None

def estimate_focal_pixels_from_exif(path, width_px):
    """
    優先順位:
    1) FocalLengthIn35mmFilm -> f_px = (f35 / 36mm) * image_width_px  （35mm判定は幅36mm仮定）
    2) FocalLength + FocalPlaneXResolution(+Unit) からピクセルピッチ経由で f_px を推定
       目安: f_px ≈ f_mm / sensor_width_mm * width_px
       ここでは FocalPlaneXResolution が [pixels per unit] なので単位に応じてピッチを算出
    """
    ex = read_exif(path)
    f35 = ex.get("FocalLengthIn35mmFilm", None)
    if f35 is not None:
        f35 = _as_float(f35)
        if f35 and f35 > 0:
            f_px = (f35 / 36.0) * width_px
            return f_px, {"source": "EXIF:FocalLengthIn35mmFilm", "f35_mm": f35}

    # 次善策: FocalLength + FocalPlaneXResolution
    f_mm = ex.get("FocalLength", None)
    f_mm = _as_float(f_mm)
    xres = ex.get("FocalPlaneXResolution", None)
    yres = ex.get("FocalPlaneYResolution", None)
    unit = ex.get("FocalPlaneResolutionUnit", None)  # 2: inch, 3: cm 等（カメラ依存）

    if f_mm and xres:
        xres = _as_float(xres)
        if xres and xres > 0:
            # 単位処理（EXIF規格: 2=inch, 3=cm, 4=mm, 5=um）
            # pixels per unit -> unit length per pixel = 1/xres [unit/pixel]
            # そこからセンサー幅[unit] ≈ image_width_px / xres
            # sensor_width_mm を得たいので、unit→mm換算
            unit_to_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}
            mm_per_unit = unit_to_mm.get(unit, 25.4)  # 不明時は inch を仮定
            sensor_width_mm = (width_px / xres) * mm_per_unit
            if sensor_width_mm > 0.0:
                f_px = (f_mm / sensor_width_mm) * width_px
                return f_px, {"source": "EXIF:FocalLength+FocalPlaneXResolution", "f_mm": f_mm, "sensor_w_mm_est": sensor_width_mm}

    return None, {"source": "fallback"}

def auto_rectify_uncalibrated(imgL, imgR, max_features=5000):
    """ 特徴点マッチ→Fundamental→stereoRectifyUncalibrated→射影整列（ホモグラフィ） """
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kL, dL = orb.detectAndCompute(grayL, None)
    kR, dR = orb.detectAndCompute(grayR, None)

    if dL is None or dR is None or len(kL) < 8 or len(kR) < 8:
        return imgL, imgR, np.eye(3), np.eye(3), None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms = bf.match(dL, dR)
    ms = sorted(ms, key=lambda m: m.distance)[:max(50, len(ms)//2)]

    ptsL = np.float32([kL[m.queryIdx].pt for m in ms])
    ptsR = np.float32([kR[m.trainIdx].pt for m in ms])

    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        return imgL, imgR, np.eye(3), np.eye(3), None

    h, w = grayL.shape
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(ptsL[mask.ravel() == 1], ptsR[mask.ravel() == 1], F, imgSize=(w, h))
    if not retval:
        return imgL, imgR, np.eye(3), np.eye(3), None

    rectL = cv2.warpPerspective(imgL, H1, (w, h))
    rectR = cv2.warpPerspective(imgR, H2, (w, h))
    return rectL, rectR, H1, H2, F

def estimate_disparity_range_by_matches(imgL, imgR, sample=3000):
    """ ORBマッチから水平シフトの分布を見て、minDisparity/numDisparities を自動案内 """
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=sample)
    kL, dL = orb.detectAndCompute(grayL, None)
    kR, dR = orb.detectAndCompute(grayR, None)
    if dL is None or dR is None or len(kL) < 8 or len(kR) < 8:
        return 0, 128  # デフォルト

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms = bf.match(dL, dR)
    if len(ms) < 20:
        return 0, 128
    ptsL = np.float32([kL[m.queryIdx].pt for m in ms])
    ptsR = np.float32([kR[m.trainIdx].pt for m in ms])
    dx = ptsL[:,0] - ptsR[:,0]  # 左→右の水平視差
    dx = dx[np.isfinite(dx)]
    if dx.size == 0:
        return 0, 128

    q1, q9 = np.quantile(dx, [0.1, 0.9])
    min_disp = int(np.floor(max(0, q1 - 4)))
    max_disp = int(np.ceil(max(q9 + 4, min_disp + 32)))
    # SGBMは16の倍数
    num_disp = int(math.ceil((max_disp - min_disp)/16.0))*16
    return min_disp, max(16, num_disp)

def build_sgbm(min_disp, num_disp, img_channels=3, block=7):
    P1 = 8*img_channels*(block**2)
    P2 = 32*img_channels*(block**2)
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=P1, P2=P2,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=1,
        disp12MaxDiff=1
    )

def lr_consistency_and_warp(left, right, dL, dR):
    dL = dL.astype(np.float32); dR = dR.astype(np.float32)
    dL[dL <= 0] = np.nan
    dR[dR <= 0] = np.nan
    H, W = dL.shape
    x = np.tile(np.arange(W)[None, :], (H, 1)).astype(np.float32)
    y = np.tile(np.arange(H)[:, None], (1, W)).astype(np.float32)

    xR_from_L = x - dL
    map_x = np.clip(xR_from_L, 0, W-1)
    map_y = y
    dR_on_L = cv2.remap(dR, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    tol = 1.5
    inconsistent = np.abs(dL + dR_on_L) > tol
    occluded_L = np.isnan(dL) | inconsistent

    xL_from_R = x + dR
    map_x_R2L = np.clip(xL_from_R, 0, W-1)
    right_warp_to_L = cv2.remap(right, map_x_R2L, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    valid_warp = ~np.isnan(dR)
    valid_warp_L = cv2.remap(valid_warp.astype(np.uint8), map_x_R2L, map_y,
                             cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)

    dR_warp = cv2.remap(dR, map_x_R2L, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    bg = left.copy()
    has_both = (~np.isnan(dL)) & valid_warp_L & (~np.isnan(dR_warp))
    choose_right = np.zeros((H, W), dtype=bool)
    choose_right[has_both] = (dR_warp[has_both] < dL[has_both])  # 視差小=遠い
    fill_from_right = occluded_L & valid_warp_L
    use_right = choose_right | fill_from_right
    bg[use_right] = right_warp_to_L[use_right]

    # 横方向伝播で実画素のみ補完
    hole = (bg.sum(axis=2) == 0)
    if hole.any():
        bg_filled = bg.copy()
        for yy in range(H):
            row = bg[yy]
            mask = hole[yy]
            if not mask.any(): continue
            last = None
            for xx in range(W):
                if not mask[xx]:
                    last = row[xx]
                elif last is not None:
                    bg_filled[yy, xx] = last
            last = None
            for xx in range(W-1, -1, -1):
                if not mask[xx]:
                    last = row[xx]
                elif last is not None:
                    bg_filled[yy, xx] = ((bg_filled[yy, xx].astype(np.uint16) + last.astype(np.uint16)) // 2)
        bg = bg_filled

    # 継ぎ目の軽い平滑
    bg = cv2.edgePreservingFilter(bg, flags=1, sigma_s=30, sigma_r=0.2)
    return bg

# ========== メイン ==========
def main(left_path, right_path, out_path="background_from_real_pixels.png"):
    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)
    assert left is not None and right is not None, "画像の読み込みに失敗しました。"

    H, W = left.shape[:2]
    print(f"[info] image size: {W}x{H}px")

    # ---- 焦点距離(ピクセル)の自動推定（任意情報）----
    f_px, meta = estimate_focal_pixels_from_exif(left_path, W)
    if f_px:
        print(f"[info] estimated focal length (pixels): {f_px:.1f}  ({meta})")
    else:
        print("[info] focal length not found in EXIF -> 処理には不要なので続行（視差レンジは特徴点から自動推定）")

    # ---- 自動整列（Uncalibrated Rectification）----
    rectL, rectR, H1, H2, F = auto_rectify_uncalibrated(left, right)
    if F is None:
        print("[warn] 自動整列が不十分。元画像のまま続行（視差はやや不安定になる可能性）")
        rectL, rectR = left, right

    # ---- 視差レンジ自動推定 → SGBM構築 ----
    min_disp, num_disp = estimate_disparity_range_by_matches(rectL, rectR)
    print(f"[info] disparity range: min={min_disp}, num={num_disp}  (SGBM要件: numは16の倍数)")

    sgbm = build_sgbm(min_disp, num_disp, img_channels=3, block=7)

    dL = sgbm.compute(rectL, rectR).astype(np.float32) / 16.0
    dR = sgbm.compute(rectR, rectL).astype(np.float32) / 16.0

    # ---- 実画素のみで背景合成 ----
    bg = lr_consistency_and_warp(rectL, rectR, dL, dR)

    cv2.imwrite(out_path, bg)
    print(f"[done] wrote: {out_path}")

if __name__ == "__main__":
    # 使い方: python stereo_real_pixels_autoparams.py left.png right.png
    import sys
    if len(sys.argv) < 3:
        print("Usage: python stereo_real_pixels_autoparams.py <left> <right> [out]")
        sys.exit(1)
    left_p = sys.argv[1]; right_p = sys.argv[2]
    out_p = sys.argv[3] if len(sys.argv) >= 4 else "background_from_real_pixels.png"
    main(left_p, right_p, out_p)