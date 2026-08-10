// DepthRaycaster.cs
//
// Takes the pointing ray from PointingRayController and figures out exactly
// which physical surface it hits, using the room mesh Meta's Scene API
// builds from the headset's depth sensors.
//
// SCAFFOLD STATUS: the actual raycast (Physics.Raycast against collider
// geometry) is standard, real Unity API and will work as written once scene
// colliders exist. What's uncertain is HOW those colliders get there: this
// script assumes Meta's Scene API (OVRSceneManager) has populated the scene
// with mesh/plane/volume anchors that have colliders attached, which is the
// documented, stable way to do physics queries against a scanned room.
// Meta's newer Depth API (EnvironmentDepthManager) is primarily a
// texture-based occlusion feature for rendering, not an exposed CPU-side
// raycast query as of this writing -- everything about that alternate path
// is marked TODO/verify below rather than assumed.
//
// Expected scene setup: attach to the same rig as PointingRayController (or
// anywhere with a reference to it), with `sceneGeometryLayer` set to
// whatever physics layer your Scene API room mesh colliders are on.

using System;
using UnityEngine;

namespace SpatialPointerVision
{
    /// <summary>Result of a successful pointing raycast against the room mesh.</summary>
    public readonly struct PointingHitResult
    {
        public readonly Vector3 WorldPoint;
        public readonly Vector3 WorldNormal;
        public readonly float Distance;
        public readonly Transform HitTransform;

        public PointingHitResult(Vector3 worldPoint, Vector3 worldNormal, float distance, Transform hitTransform)
        {
            WorldPoint = worldPoint;
            WorldNormal = worldNormal;
            Distance = distance;
            HitTransform = hitTransform;
        }
    }

    /// <summary>
    /// Raycasts the pointing ray against the Scene API's room mesh to find the
    /// exact 3D point (and underlying object) the user is pointing at.
    /// </summary>
    [RequireComponent(typeof(PointingRayController))]
    public class DepthRaycaster : MonoBehaviour
    {
        [Header("Dependencies")]
        [SerializeField]
        private PointingRayController pointingRayController;

        [Header("Raycast settings")]
        [Tooltip("Physics layer(s) the Scene API room mesh colliders live on. " +
                 "TODO: verify against current Meta XR SDK docs -- confirm which " +
                 "layer OVRSceneManager-spawned anchors use by default, or assign " +
                 "one explicitly when configuring the Scene prefab.")]
        [SerializeField]
        private LayerMask sceneGeometryLayer = ~0;

        [SerializeField]
        private float maxPointingDistance = 5f;

        /// <summary>Raised every frame the pointing ray hits scene geometry.</summary>
        public event Action<PointingHitResult> OnObjectHit;

        /// <summary>Raised when the pointing ray stops hitting anything (or stops being valid).</summary>
        public event Action OnHitLost;

        private bool _hadHitLastFrame;

        private void Reset()
        {
            pointingRayController = GetComponent<PointingRayController>();
        }

        private void OnEnable()
        {
            if (pointingRayController != null)
            {
                pointingRayController.OnPointingRayUpdated += HandlePointingRayUpdated;
                pointingRayController.OnPointingEnded += HandlePointingEnded;
            }
        }

        private void OnDisable()
        {
            if (pointingRayController != null)
            {
                pointingRayController.OnPointingRayUpdated -= HandlePointingRayUpdated;
                pointingRayController.OnPointingEnded -= HandlePointingEnded;
            }
        }

        private void HandlePointingRayUpdated(Ray ray)
        {
            // Standard Unity physics query -- this part is not Meta-SDK-specific
            // and will work exactly as written, PROVIDED the room mesh actually
            // has colliders on `sceneGeometryLayer`. See TODOs above/below for
            // how those colliders are expected to get there.
            bool didHit = Physics.Raycast(
                ray,
                out RaycastHit hit,
                maxPointingDistance,
                sceneGeometryLayer,
                QueryTriggerInteraction.Ignore);

            if (!didHit)
            {
                if (_hadHitLastFrame)
                {
                    _hadHitLastFrame = false;
                    OnHitLost?.Invoke();
                }
                return;
            }

            _hadHitLastFrame = true;
            var result = new PointingHitResult(hit.point, hit.normal, hit.distance, hit.transform);
            OnObjectHit?.Invoke(result);
        }

        private void HandlePointingEnded()
        {
            if (_hadHitLastFrame)
            {
                _hadHitLastFrame = false;
                OnHitLost?.Invoke();
            }
        }

        // TODO: verify against current Meta XR SDK docs. Alternative approach
        // worth investigating once you're in the actual Unity/Meta XR SDK
        // environment: EnvironmentDepthManager exposes the live depth texture
        // used for occlusion rendering. If a CPU-readable depth sample API
        // exists on the SDK version you install, sampling actual per-pixel
        // depth at the pointing ray's screen-space projection would be more
        // accurate than colliders for objects the Scene API's room-scan
        // didn't capture as a distinct anchor (e.g. small objects on a
        // table). This is NOT implemented here because I could not verify
        // the exact API surface without the SDK installed -- confirm what's
        // available before relying on it.
    }
}
