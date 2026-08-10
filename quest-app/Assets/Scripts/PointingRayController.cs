// PointingRayController.cs
//
// Reads the user's hand-tracking pose from the Meta XR SDK and exposes it as
// a simple world-space ray (origin + direction) that DepthRaycaster consumes
// to figure out what's being pointed at.
//
// SCAFFOLD STATUS: method signatures + control flow are real; the Meta XR
// SDK member names are written from what's documented for the Oculus
// Integration / Meta XR Core SDK's OVRHand component, but SDK surfaces shift
// between versions -- every line touching OVRHand is marked
// "// TODO: verify against current Meta XR SDK docs" and should be checked
// against whatever SDK version you actually install, since this couldn't be
// compiled or tested without Unity + the SDK present.
//
// Expected scene setup: attach to a GameObject under OVRCameraRig, with
// `hand` pointing at the OVRHand component on the tracked hand you want to
// use for pointing (typically the right hand).

using System;
using UnityEngine;

namespace SpatialPointerVision
{
    /// <summary>
    /// Tracks the user's pointing gesture and raises <see cref="OnPointingRayUpdated"/>
    /// every frame a valid pointing ray is available.
    /// </summary>
    public class PointingRayController : MonoBehaviour
    {
        [Header("Hand tracking source")]
        [Tooltip("The OVRHand component to read the pointing pose from (usually the right hand).")]
        [SerializeField]
        private OVRHand hand; // TODO: verify against current Meta XR SDK docs (OVRHand class + assembly reference)

        [Header("Gesture thresholds")]
        [Tooltip("Minimum hand tracking confidence required before we trust the pointing ray.")]
        [SerializeField]
        private OVRHand.TrackingConfidence minConfidence = OVRHand.TrackingConfidence.High; // TODO: verify against current Meta XR SDK docs

        [Tooltip("How long (seconds) the pointing pose must be valid before we start firing updates, to avoid spurious single-frame flicker.")]
        [SerializeField]
        private float poseStabilizationSeconds = 0.1f;

        /// <summary>Raised every frame a stable pointing ray is available.</summary>
        public event Action<Ray> OnPointingRayUpdated;

        /// <summary>Raised when the pointing gesture starts (goes from invalid/unstable to stable).</summary>
        public event Action OnPointingStarted;

        /// <summary>Raised when the pointing gesture ends (hand no longer tracked / low confidence).</summary>
        public event Action OnPointingEnded;

        /// <summary>The most recent stable pointing ray, in world space.</summary>
        public Ray CurrentPointingRay { get; private set; }

        /// <summary>True if <see cref="CurrentPointingRay"/> was updated this frame.</summary>
        public bool HasValidPointingRay { get; private set; }

        private float _stableSince = -1f;
        private bool _wasPointing;

        private void Reset()
        {
            hand = GetComponentInChildren<OVRHand>(); // TODO: verify against current Meta XR SDK docs
        }

        private void Update()
        {
            if (hand == null)
            {
                SetPointing(false);
                return;
            }

            // OVRHand exposes hand tracking confidence, whether tracking is
            // currently valid at all, and a "pointer pose" transform derived
            // from the hand skeleton that's the SDK's own best estimate of
            // where the user is "pointing" (used internally for UI ray
            // interaction). Reusing it here rather than re-deriving a
            // pointing direction from raw joint transforms keeps this
            // consistent with how Meta's own UI raycasting behaves.
            // TODO: verify against current Meta XR SDK docs -- confirm
            // IsTracked / IsPointerPoseValid / HandConfidence / PointerPose
            // are still the correct member names on the SDK version you
            // install, and confirm PointerPose is the right source for a
            // "point at a physical object" gesture vs. e.g. index fingertip
            // + a joint further back on the hand for a more intuitive aim.
            bool trackedWithConfidence =
                hand.IsTracked
                && hand.IsPointerPoseValid
                && hand.HandConfidence >= minConfidence;

            if (!trackedWithConfidence)
            {
                _stableSince = -1f;
                SetPointing(false);
                return;
            }

            if (_stableSince < 0f)
            {
                _stableSince = Time.time;
            }

            bool stableEnough = Time.time - _stableSince >= poseStabilizationSeconds;
            if (!stableEnough)
            {
                SetPointing(false);
                return;
            }

            Transform pointerPose = hand.PointerPose; // TODO: verify against current Meta XR SDK docs
            var ray = new Ray(pointerPose.position, pointerPose.forward);

            CurrentPointingRay = ray;
            HasValidPointingRay = true;
            SetPointing(true);
            OnPointingRayUpdated?.Invoke(ray);
        }

        private void SetPointing(bool isPointing)
        {
            HasValidPointingRay = isPointing;
            if (isPointing == _wasPointing)
            {
                return;
            }

            _wasPointing = isPointing;
            if (isPointing)
            {
                OnPointingStarted?.Invoke();
            }
            else
            {
                OnPointingEnded?.Invoke();
            }
        }
    }
}
