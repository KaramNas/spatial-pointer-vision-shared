// ObjectCropper.cs
//
// Given a 3D world point the user is pointing at (from DepthRaycaster), crops
// a small region of the passthrough camera's video feed around that point
// and hands the result off as JPEG bytes for InferenceClient to send to the
// server.
//
// SCAFFOLD STATUS: the crop/encode math (Texture2D pixel extraction, JPEG
// encoding) is real, standard Unity API. The part that's genuinely uncertain
// is projecting a 3D world point into the PHYSICAL passthrough camera's 2D
// pixel space -- that is NOT the same camera as Unity's rendering eye
// cameras. The passthrough video comes from separate RGB sensors on the
// headset with their own intrinsics/extrinsics, so a plain
// Camera.WorldToScreenPoint() call against a render camera will project to
// the wrong pixels. Meta's Passthrough Camera API was built specifically to
// expose those sensor intrinsics/extrinsics for this exact "point at
// something in the real world, find it in the camera image" use case, so
// the real accessor almost certainly exists -- but I couldn't verify its
// exact name without the SDK installed, so it's marked TODO below rather
// than guessed at.
//
// Expected scene setup: attach anywhere with a reference to DepthRaycaster.
// `passthroughCameraTexture` should be assigned to whatever WebCamTexture
// (or equivalent) the Meta Passthrough Camera API hands you at runtime.

using System;
using UnityEngine;

namespace SpatialPointerVision
{
    /// <summary>
    /// Crops the passthrough camera feed around a pointed-at world point.
    /// </summary>
    [RequireComponent(typeof(DepthRaycaster))]
    public class ObjectCropper : MonoBehaviour
    {
        [Header("Dependencies")]
        [SerializeField]
        private DepthRaycaster depthRaycaster;

        [Tooltip("The live passthrough camera feed. TODO: verify against current " +
                 "Meta XR SDK docs -- assign this from whatever component the " +
                 "Passthrough Camera API exposes (e.g. a WebCamTextureManager-style " +
                 "component in the Meta sample repo), not a generic webcam.")]
        [SerializeField]
        private WebCamTexture passthroughCameraTexture;

        [Header("Crop settings")]
        [SerializeField]
        [Tooltip("Side length, in pixels, of the (square) crop region before any distance scaling.")]
        private int baseCropSizePixels = 384;

        [SerializeField]
        [Tooltip("Minimum crop size regardless of distance, so far-away objects don't shrink to nothing.")]
        private int minCropSizePixels = 128;

        [SerializeField]
        [Range(1, 100)]
        private int jpegQuality = 85;

        [SerializeField]
        [Tooltip("Minimum time between captures, to avoid saturating the inference server with every frame.")]
        private float captureCooldownSeconds = 1.0f;

        /// <summary>Raised when a crop has been produced and is ready to send for inference.</summary>
        public event Action<byte[]> OnObjectCropped;

        private float _lastCaptureTime = float.NegativeInfinity;

        private void Reset()
        {
            depthRaycaster = GetComponent<DepthRaycaster>();
        }

        private void OnEnable()
        {
            if (depthRaycaster != null)
            {
                depthRaycaster.OnObjectHit += HandleObjectHit;
            }
        }

        private void OnDisable()
        {
            if (depthRaycaster != null)
            {
                depthRaycaster.OnObjectHit -= HandleObjectHit;
            }
        }

        private void HandleObjectHit(PointingHitResult hit)
        {
            if (Time.time - _lastCaptureTime < captureCooldownSeconds)
            {
                return;
            }

            if (passthroughCameraTexture == null || !passthroughCameraTexture.didUpdateThisFrame)
            {
                return;
            }

            if (!TryProjectWorldPointToCameraPixel(hit.WorldPoint, out Vector2Int pixelCenter))
            {
                return;
            }

            int cropSize = Mathf.Max(minCropSizePixels, ScaleCropSizeByDistance(hit.Distance));
            byte[] jpegBytes = CropAndEncode(passthroughCameraTexture, pixelCenter, cropSize);
            if (jpegBytes == null)
            {
                return;
            }

            _lastCaptureTime = Time.time;
            OnObjectCropped?.Invoke(jpegBytes);
        }

        private int ScaleCropSizeByDistance(float distanceMeters)
        {
            // Closer objects fill more of the frame -> a smaller crop region
            // captures them tightly; farther objects need a larger region to
            // make sure the object is actually inside the box at all. This
            // linear falloff is a starting point, not a calibrated model --
            // tune against real passthrough footage once available.
            const float referenceDistance = 1.0f; // meters
            float scale = distanceMeters / referenceDistance;
            return Mathf.RoundToInt(baseCropSizePixels * Mathf.Clamp(scale, 0.5f, 2.5f));
        }

        /// <summary>
        /// Projects a 3D world point into the passthrough camera's 2D pixel
        /// space. TODO: verify against current Meta XR SDK docs -- this
        /// should use the Passthrough Camera API's actual sensor intrinsics
        /// (focal length, principal point, distortion, and the camera's own
        /// tracked pose) rather than a generic Unity camera transform, since
        /// the passthrough sensors are physically offset from and have
        /// different optical properties than Unity's rendering eye cameras.
        /// The naive fallback below (using `fallbackProjectionCamera`, a
        /// plain Unity Camera) is only correct for quick Editor testing with
        /// a regular webcam standing in for passthrough -- do not ship it
        /// as-is.
        /// </summary>
        private bool TryProjectWorldPointToCameraPixel(Vector3 worldPoint, out Vector2Int pixelCenter)
        {
            pixelCenter = default;

            if (fallbackProjectionCamera == null)
            {
                Debug.LogWarning(
                    "[ObjectCropper] No camera projection source configured. " +
                    "TODO: wire up the Passthrough Camera API's intrinsics here.");
                return false;
            }

            Vector3 screenPoint = fallbackProjectionCamera.WorldToScreenPoint(worldPoint);
            if (screenPoint.z <= 0f)
            {
                return false; // behind the camera
            }

            int px = Mathf.RoundToInt(screenPoint.x / fallbackProjectionCamera.pixelWidth * passthroughCameraTexture.width);
            int py = Mathf.RoundToInt(screenPoint.y / fallbackProjectionCamera.pixelHeight * passthroughCameraTexture.height);
            pixelCenter = new Vector2Int(px, py);
            return true;
        }

        [Header("Editor/testing fallback only -- see TODO in TryProjectWorldPointToCameraPixel")]
        [SerializeField]
        private Camera fallbackProjectionCamera;

        private Texture2D _scratchTexture; // reused across captures to avoid per-frame allocations

        private byte[] CropAndEncode(WebCamTexture source, Vector2Int pixelCenter, int cropSize)
        {
            int half = cropSize / 2;
            int x = Mathf.Clamp(pixelCenter.x - half, 0, Mathf.Max(0, source.width - cropSize));
            int y = Mathf.Clamp(pixelCenter.y - half, 0, Mathf.Max(0, source.height - cropSize));
            int width = Mathf.Min(cropSize, source.width - x);
            int height = Mathf.Min(cropSize, source.height - y);

            if (width <= 0 || height <= 0)
            {
                return null;
            }

            if (_scratchTexture == null || _scratchTexture.width != width || _scratchTexture.height != height)
            {
                if (_scratchTexture != null)
                {
                    Destroy(_scratchTexture);
                }
                _scratchTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
            }

            Color32[] pixels = source.GetPixels32();
            var cropped = new Color32[width * height];
            for (int row = 0; row < height; row++)
            {
                int srcRowStart = (y + row) * source.width + x;
                int dstRowStart = row * width;
                Array.Copy(pixels, srcRowStart, cropped, dstRowStart, width);
            }

            _scratchTexture.SetPixels32(cropped);
            _scratchTexture.Apply(false);

            return ImageConversion.EncodeToJPG(_scratchTexture, jpegQuality);
        }

        private void OnDestroy()
        {
            if (_scratchTexture != null)
            {
                Destroy(_scratchTexture);
            }
        }
    }
}
