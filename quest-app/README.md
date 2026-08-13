# Quest 3 client (Unity)

Not runnable yet in this environment -- there's no Unity project here, just
the C# scripts that will become one. This document covers what you need to
install and wire up before that's true.

## Why this is scaffold-only

Building and testing this piece requires:
- Unity Hub + **Unity 6** (not 2022 LTS -- see version note below)
- The **Meta XR SDK** (Package Manager: Core, Interaction SDK) via the
  **Unity OpenXR Plugin**
- A **Meta Quest 3** to actually test hand tracking / passthrough / depth on
  (the Editor can simulate some of this but not reliably enough to trust)

**Version note (checked against Meta's docs, not guessed):** this project
was originally scoped against Unity 2022 LTS + the older Oculus XR Plugin,
which is now deprecated. Meta's current recommended path is the **Unity
OpenXR Plugin**, which requires **Unity 6+** and **Meta XR SDK v74+** --
and critically, **the Depth API that `DepthRaycaster.cs` depends on is not
supported at all on Unity versions before 6**. So Unity 6 isn't just
"newer and fine," it's required for this specific project's room-depth
raycasting to work.

None of that exists in this repo's dev environment, so these scripts were
written as realistic, XML-documented method signatures with the actual logic
filled in wherever it's standard Unity API (raycasting, texture cropping,
JSON parsing, UI), and `// TODO: verify against current Meta XR SDK docs`
comments anywhere the exact Meta XR SDK member name is uncertain -- rather
than inventing a plausible-looking call that might not exist on whatever SDK
version you install.

## Scripts and how they connect

```
PointingRayController  (hand tracking -> world-space pointing ray)
        |  OnPointingRayUpdated(Ray)
        v
DepthRaycaster          (ray -> Scene API room-mesh raycast -> world hit point)
        |  OnObjectHit(PointingHitResult)
        v
ObjectCropper           (world point -> passthrough camera pixel -> JPEG crop)
        |  OnObjectCropped(byte[])
        v
InferenceClient         (JPEG bytes --ws--> server/main.py --ws--> label+confidence)
        |  OnResultReceived(IdentifyResult) / OnInferenceError(string)
        v
ResultOverlay            (label+confidence -> floating world-space UI)
```

Each script only knows about the one upstream it depends on (enforced via
`[RequireComponent]` + C# events) -- there's no single "god" controller
script wiring all five together. A thin scene-specific glue script (or just
Inspector-wired `UnityEvent`s, or a few `AddListener` calls in a scene
bootstrap `MonoBehaviour`) connects `ObjectCropper.OnObjectCropped` to
`InferenceClient.SendImageAsync`, and `InferenceClient.OnResultReceived` /
`OnInferenceError` to `ResultOverlay.ShowResult` / `ShowError`. That glue
script isn't included since its shape depends on your actual scene
hierarchy.

## Setup

1. Install **Unity Hub**, then **Unity 6** through it (Android Build
   Support module included). If a Unity 6.x Editor with Android + Windows
   support is already installed on your machine, you likely don't need to
   install anything here -- just confirm the Android module is present
   (Unity Hub > Installs > your version > gear icon > Add Modules).
2. Create a new 3D (URP recommended) project using that Unity 6 install, or
   open one if you've already started this part.
3. **Window > Package Manager > Unity Registry**, install **OpenXR Plugin**
   (this is what replaces the old Oculus XR Plugin path).
4. **Window > Package Manager > Add package from git URL**:
   - `https://github.com/endel/NativeWebSocket.git#upm` (required by
     `InferenceClient.cs` -- see the comment at the top of that file for why
     this was chosen over `System.Net.WebSockets`)
5. Install the **Meta XR SDK v74+** via the Unity Asset Store or Package
   Manager (search "Meta XR"): at minimum the **Meta XR Core SDK** (hand
   tracking, `OVRCameraRig`, Scene/Depth API) and **Meta XR Interaction
   SDK**. Follow Meta's own project setup tool (Meta > Tools > Project
   Setup Tool in the Unity menu once installed) -- it auto-configures most
   of the Android/XR player settings this needs, and will guide you through
   enabling OpenXR + the Meta Quest feature group under **Project Settings
   > XR Plug-in Management**.
6. **Window > TextMeshPro > Import TMP Essential Resources** (needed by
   `ResultOverlay.cs`).
7. Copy/import this repo's `quest-app/Assets/Scripts/` into your project's
   `Assets/Scripts/` if you started the Unity project outside this repo, or
   point Unity at this repo directly if you created the project here (in
   which case they're already in the right place).
8. Enable required permissions/capabilities in **Meta > Tools > Project
   Setup Tool** and the Android manifest: hand tracking, passthrough, scene
   (room mesh), and camera access. Exact toggle names vary by SDK version --
   the Project Setup Tool flags anything missing.
9. Build an `OVRCameraRig` in your scene (Meta's prefab), add
   `PointingRayController` + `DepthRaycaster` + `ObjectCropper` to it (or a
   child object), wire `PointingRayController.hand` to the rig's right-hand
   `OVRHand`, and set up the Scene API (`OVRSceneManager`) so room-mesh
   anchors get colliders for `DepthRaycaster` to hit.
10. Add `InferenceClient` and `ResultOverlay` anywhere in the scene; set
    `InferenceClient.serverHost` to your PC's LAN IP (shown when you run
    `python server/main.py` -- see the root README).
11. Build for Android (File > Build Settings > Android > Switch Platform),
    deploy to a Quest 3 over USB (Developer Mode enabled on the headset).

**Suggestion for a tight disk budget:** if C: is low on space, point Unity
Hub's install location and your new project's location at another drive
with room (Unity Hub > Preferences > lets you change the default Editor/
project install location) -- Unity projects with Android build support can
easily use 10-20GB between the Editor, Android SDK/NDK components, and
Library/ build cache.

## Two things to get right before trusting results

- **Passthrough camera projection** (`ObjectCropper.cs`): mapping a 3D world
  point into the *physical* passthrough camera's 2D pixel space needs that
  camera's real sensor intrinsics/extrinsics from the Passthrough Camera
  API, not a generic Unity `Camera.WorldToScreenPoint()` call against a
  render camera (different sensor, different optics). Flagged clearly with
  a TODO in the code -- this is the single most important thing to verify
  and fix before the crop sent to the server actually contains the object
  the user pointed at.
- **Room-mesh collider coverage** (`DepthRaycaster.cs`): raycasting only
  finds what the Scene API's room scan captured as distinct geometry. Small
  objects sitting on a scanned table surface may not be separately
  represented -- test against real pointing gestures at real objects before
  assuming hit accuracy.
