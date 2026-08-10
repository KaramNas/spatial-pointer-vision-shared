// InferenceClient.cs
//
// WebSocket client that connects to the FastAPI server's /ws/identify
// endpoint (see server/main.py), sends cropped-object JPEG bytes from
// ObjectCropper, and surfaces the returned label/confidence (or error) to
// listeners such as ResultOverlay.
//
// Uses the NativeWebSocket package rather than System.Net.WebSockets, since
// System.Net.WebSockets.ClientWebSocket has a history of being unreliable
// under IL2CPP on Android (which is what Quest builds use) across Unity
// versions, while NativeWebSocket (github.com/endel/NativeWebSocket) is the
// community-standard cross-platform choice that also works in WebGL. This is
// NOT bundled with Unity -- add it via Package Manager -> "Add package from
// git URL" -> https://github.com/endel/NativeWebSocket.git#upm before this
// file will compile. See quest-app/README.md.
//
// SCAFFOLD STATUS: this is a real, complete WebSocket client implementation
// against NativeWebSocket's public API. It has not been compiled (no Unity
// project exists yet in this environment) -- double check it against
// whatever NativeWebSocket package version you actually install, flagged
// inline where relevant.

using System;
using System.Text;
using System.Threading.Tasks;
using NativeWebSocket;
using UnityEngine;

namespace SpatialPointerVision
{
    /// <summary>
    /// Connects to the tethered inference server and exchanges cropped image
    /// bytes for object-identification results.
    /// </summary>
    public class InferenceClient : MonoBehaviour
    {
        [Header("Server connection")]
        [Tooltip("LAN IP of the PC running `python server/main.py` (printed at server startup).")]
        [SerializeField]
        private string serverHost = "192.168.1.100";

        [SerializeField]
        private int serverPort = 8000;

        [SerializeField]
        private string endpointPath = "/ws/identify";

        [Header("Reconnection")]
        [SerializeField]
        private float reconnectDelaySeconds = 3f;

        [SerializeField]
        private bool autoReconnect = true;

        /// <summary>Successful identification result.</summary>
        [Serializable]
        public struct IdentifyResult
        {
            public string label;
            public float confidence;
        }

        /// <summary>Raised when the server returns a successful identification.</summary>
        public event Action<IdentifyResult> OnResultReceived;

        /// <summary>Raised on any error: malformed response, server-side error, or connection failure.</summary>
        public event Action<string> OnInferenceError;

        public bool IsConnected => _ws != null && _ws.State == WebSocketState.Open;

        // Mirrors the JSON shapes server/main.py's /ws/identify can send:
        // either {"label": ..., "confidence": ...} or {"error": ..., "message": ...}.
        [Serializable]
        private class ServerMessage
        {
            public string label;
            public float confidence;
            public string error;
            public string message;
        }

        private WebSocket _ws;
        private bool _quitting;

        private async void Start()
        {
            await ConnectAsync();
        }

        public async Task ConnectAsync()
        {
            string url = $"ws://{serverHost}:{serverPort}{endpointPath}";
            _ws = new WebSocket(url);

            _ws.OnOpen += () => Debug.Log($"[InferenceClient] Connected to {url}");
            _ws.OnMessage += HandleMessage;
            _ws.OnError += HandleSocketError;
            _ws.OnClose += HandleSocketClose;

            try
            {
                await _ws.Connect();
            }
            catch (Exception e)
            {
                Debug.LogError($"[InferenceClient] Failed to connect to {url}: {e.Message}");
                OnInferenceError?.Invoke($"Could not connect to inference server at {url}.");
                if (autoReconnect && !_quitting)
                {
                    _ = RetryConnectAfterDelay();
                }
            }
        }

        /// <summary>Sends a cropped JPEG frame for identification. Fire-and-forget from
        /// ObjectCropper.OnObjectCropped is the expected usage pattern.</summary>
        public async Task SendImageAsync(byte[] jpegBytes)
        {
            if (!IsConnected)
            {
                Debug.LogWarning("[InferenceClient] Not connected; dropping frame.");
                return;
            }

            // TODO: verify against current NativeWebSocket docs for the
            // installed package version -- `Send` is documented as the
            // binary-frame send method as of the versions I'm aware of, with
            // `SendText` as the separate text-frame method. Confirm this
            // still matches once the package is actually installed.
            await _ws.Send(jpegBytes);
        }

        private void HandleMessage(byte[] rawBytes)
        {
            string json = Encoding.UTF8.GetString(rawBytes);

            ServerMessage parsed;
            try
            {
                parsed = JsonUtility.FromJson<ServerMessage>(json);
            }
            catch (Exception e)
            {
                Debug.LogError($"[InferenceClient] Failed to parse server response: {e.Message}\nRaw: {json}");
                OnInferenceError?.Invoke("Received a malformed response from the server.");
                return;
            }

            if (!string.IsNullOrEmpty(parsed.error))
            {
                Debug.LogWarning($"[InferenceClient] Server reported error '{parsed.error}': {parsed.message}");
                OnInferenceError?.Invoke(parsed.message);
                return;
            }

            OnResultReceived?.Invoke(new IdentifyResult { label = parsed.label, confidence = parsed.confidence });
        }

        private void HandleSocketError(string errorMsg)
        {
            Debug.LogError($"[InferenceClient] WebSocket error: {errorMsg}");
            OnInferenceError?.Invoke($"Connection error: {errorMsg}");
        }

        private async void HandleSocketClose(WebSocketCloseCode closeCode)
        {
            Debug.Log($"[InferenceClient] Connection closed: {closeCode}");
            if (autoReconnect && !_quitting)
            {
                await RetryConnectAfterDelay();
            }
        }

        private async Task RetryConnectAfterDelay()
        {
            await Task.Delay(TimeSpan.FromSeconds(reconnectDelaySeconds));
            if (!_quitting)
            {
                await ConnectAsync();
            }
        }

        private void Update()
        {
            // NativeWebSocket queues incoming messages off the main thread on
            // non-WebGL platforms (Android/Quest included) and requires this
            // pump to dispatch OnMessage/OnOpen/OnClose callbacks on the main
            // thread. WebGL builds route through browser JS callbacks instead
            // and don't need it.
            // TODO: verify against current NativeWebSocket docs for the
            // installed package version.
#if !UNITY_WEBGL || UNITY_EDITOR
            _ws?.DispatchMessageQueue();
#endif
        }

        private async void OnApplicationQuit()
        {
            _quitting = true;
            if (_ws != null && _ws.State == WebSocketState.Open)
            {
                await _ws.Close();
            }
        }
    }
}
