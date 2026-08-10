// ResultOverlay.cs
//
// Displays the identified object's label + confidence as a world-space UI
// element floating near the point the user pointed at, and fades it out
// after a few seconds. Also used to surface InferenceClient errors
// (connection lost, model not loaded, etc.) in a way the user can actually
// see while wearing the headset.
//
// SCAFFOLD STATUS: this one is low-risk -- it's built entirely from stable,
// well-documented Unity/TextMeshPro APIs, nothing Meta-SDK-specific. The one
// thing worth double-checking is `facingCamera`: it should be assigned to
// OVRCameraRig's CenterEyeAnchor so the label always faces the user, but the
// exact Transform to grab depends on your rig setup.
//
// Requires TextMeshPro (Window > TextMeshPro > Import TMP Essential
// Resources -- a one-time per-project import, bundled with Unity 2022 LTS).
//
// Expected scene setup: a world-space Canvas with a TextMeshProUGUI child,
// scaled small (e.g. 0.001 per unit) so it reads as a normal-sized label at
// arm's length. Starts hidden; call ShowResult/ShowError to display it.

using System.Collections;
using TMPro;
using UnityEngine;

namespace SpatialPointerVision
{
    [RequireComponent(typeof(Canvas))]
    public class ResultOverlay : MonoBehaviour
    {
        [Header("UI")]
        [SerializeField]
        private TextMeshProUGUI labelText;

        [Header("Placement")]
        [Tooltip("Usually OVRCameraRig's CenterEyeAnchor, so the label billboard faces the user.")]
        [SerializeField]
        private Transform facingCamera; // TODO: verify against current Meta XR SDK docs -- confirm CenterEyeAnchor is still the right transform to face on your rig.

        [SerializeField]
        private float worldSpaceOffsetUp = 0.05f;

        [Header("Timing")]
        [SerializeField]
        private float displaySeconds = 2.5f;

        private Coroutine _hideRoutine;

        private void Awake()
        {
            if (labelText == null)
            {
                labelText = GetComponentInChildren<TextMeshProUGUI>();
            }
            gameObject.SetActive(false);
        }

        /// <summary>Shows a successful identification result floating above the pointed-at point.</summary>
        public void ShowResult(Vector3 worldPosition, string label, float confidence)
        {
            labelText.color = Color.white;
            labelText.text = $"{label} ({confidence:P0})";
            Display(worldPosition);
        }

        /// <summary>Shows an error/status message (e.g. "connection lost", "model not loaded").</summary>
        public void ShowError(string message)
        {
            labelText.color = Color.yellow;
            labelText.text = message;
            // Errors aren't tied to a specific world point (e.g. a dropped
            // connection can happen with no current hit) -- show in front of
            // the user instead.
            Vector3 fallbackPosition = facingCamera != null
                ? facingCamera.position + facingCamera.forward * 0.5f
                : transform.position;
            Display(fallbackPosition);
        }

        private void Display(Vector3 worldPosition)
        {
            transform.position = worldPosition + Vector3.up * worldSpaceOffsetUp;

            if (facingCamera != null)
            {
                Vector3 lookDirection = transform.position - facingCamera.position;
                if (lookDirection.sqrMagnitude > 0.0001f)
                {
                    transform.rotation = Quaternion.LookRotation(lookDirection);
                }
            }

            gameObject.SetActive(true);

            if (_hideRoutine != null)
            {
                StopCoroutine(_hideRoutine);
            }
            _hideRoutine = StartCoroutine(HideAfterDelay());
        }

        private IEnumerator HideAfterDelay()
        {
            yield return new WaitForSeconds(displaySeconds);
            gameObject.SetActive(false);
            _hideRoutine = null;
        }
    }
}
