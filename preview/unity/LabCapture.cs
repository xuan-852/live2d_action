using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Live2D.Cubism.Core;
using Live2D.Cubism.Framework.Json;
using Live2D.Cubism.Rendering;
using Live2D.Cubism.Rendering.Masking;
using Live2D.Cubism.Framework.Physics;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Editor-only synchronous baseline. No wall-clock-driven simulation or old prefabs.
public static class LabCapture
{
    [Serializable] public class ParameterState { public string id; public float value, minimum, maximum, defaultValue; }
    [Serializable] public class PartState { public string id; public float opacity; }
    [Serializable] public class DrawableState
    {
        public string id; public Vector3[] vertices; public int[] indices;
        public float opacity; public int sortingOrder; public bool enabled;
    }
    [Serializable] public class FrameState
    {
        public int frame; public double actionTime; public string wallUtc;
        public double renderReadbackMs; public string physics = "disabled; no physics stepping";
        public string[] writers; public ParameterState[] requested, applied;
        public PartState[] parts; public DrawableState[] drawables;
    }
    static object Load(Type type, string path)
    {
        if (type == typeof(byte[])) return File.ReadAllBytes(path);
        if (type == typeof(string)) return File.ReadAllText(path);
        if (type == typeof(Texture2D))
        {
            var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            if (!texture.LoadImage(File.ReadAllBytes(path))) throw new IOException(path);
            return texture;
        }
        throw new NotSupportedException(type.FullName);
    }
    static ParameterState[] Parameters(CubismModel model) => model.Parameters.Select(p => new ParameterState {
        id = p.Id, value = p.Value, minimum = p.MinimumValue, maximum = p.MaximumValue, defaultValue = p.DefaultValue
    }).ToArray();

    static AnimationClip MakeGesture(CubismModel model)
    {
        var clip = new AnimationClip { name = "turn-nod-return" };
        Action<string, float[]> curve = (id, values) => {
            var parameter = model.Parameters.Single(p => p.Id == id);
            float[] times = { 0, 0.5f, 1.3f, 1.8f, 2.2f, 2.65f, 3.2f, 4.0f };
            var keys = times.Select((t, i) => new Keyframe(t, values[i], 0, 0)).ToArray();
            clip.SetCurve(AnimationUtility.CalculateTransformPath(parameter.transform, model.transform),
                typeof(CubismParameter), "Value", new AnimationCurve(keys));
        };
        if (Environment.GetEnvironmentVariable("ACTION_LAB_MODE") == "body-turn")
        {
            curve("ParamBodyAngleX2", new float[] { 0, 0, -8, 0, 0, 8, 0, 0 });
            return clip;
        }
        curve("ParamAngleX", new float[] { 0, 0, 12, 12, 10, 8, 0, 0 });
        curve("ParamAngleY", new float[] { 0, 0, 2, 2, -10, 2, 0, 0 });
        curve("ParamAngleZ", new float[] { 0, 0, -3, -3, -2, -1, 0, 0 });
        if (Environment.GetEnvironmentVariable("ACTION_LAB_MODE") == "coordinated")
        {
            var parameter = model.Parameters.Single(p => p.Id == "ParamBodyAngleX2");
            var keys = new[] { new Keyframe(0, 0), new Keyframe(0.75f, 0),
                new Keyframe(1.6f, 7), new Keyframe(2.45f, 7),
                new Keyframe(3.45f, 0), new Keyframe(4, 0) };
            clip.SetCurve(AnimationUtility.CalculateTransformPath(parameter.transform, model.transform),
                typeof(CubismParameter), "Value", new AnimationCurve(keys));
            curve("ParamEyeBallX", new float[] { 0, 0.2f, 0.3f, 0.3f, 0.2f, 0, 0, 0 });
            return clip;
        }
        bool bodyOff = Environment.GetEnvironmentVariable("ACTION_LAB_BODY") == "off";
        curve("ParamBodyAngleX", bodyOff ? new float[8] : new float[] { 0, 0, 7, 7, 5, 3, 0, 0 });
        curve("ParamBodyAngleZ", bodyOff ? new float[8] : new float[] { 0, 0, -4, -4, -3, -1, 0, 0 });
        curve("ParamEyeBallX", new float[] { 0, 0.2f, 0.3f, 0.3f, 0.2f, 0, 0, 0 });
        return clip;
    }

    public static void Run()
    {
        try { Capture(); EditorApplication.Exit(0); }
        catch (Exception e) { UnityEngine.Debug.LogException(e); EditorApplication.Exit(1); }
    }
    static void Capture()
    {
        string modelPath = Environment.GetEnvironmentVariable("ACTION_LAB_MODEL");
        string output = Environment.GetEnvironmentVariable("ACTION_LAB_OUTPUT");
        if (string.IsNullOrEmpty(modelPath) || string.IsNullOrEmpty(output)) throw new ArgumentException("Set ACTION_LAB_MODEL and ACTION_LAB_OUTPUT");
        if (Directory.Exists(output)) throw new IOException("Refusing to overwrite run: " + output);
        Directory.CreateDirectory(output);
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var model = CubismModel3Json.LoadAtPath(modelPath, Load).ToModel();
        if (model == null) throw new Exception("Model creation failed");
        // All automatic writers, including Animator, eye blink and physics, are disabled.
        var writers = model.GetComponentsInChildren<Behaviour>().Select(b => b.GetType().FullName + ":disabled").ToArray();
        foreach (var b in model.GetComponentsInChildren<Behaviour>()) b.enabled = false;
        var render = model.GetComponent<CubismRenderController>();
        render.enabled = true;
        render.TryInitializeRenderers();
        var masks = model.GetComponent<CubismMaskController>();
        if (masks != null) masks.enabled = true;
        foreach (var p in model.Parameters) p.Value = p.DefaultValue;
        model.ForceUpdateNow();
        render.OnLateUpdate();
        if (masks != null) masks.OnLateUpdate();
        var bounds = new Bounds();
        bool first = true;
        foreach (var r in render.Renderers.Where(r => r.Mesh.colors.Any(c => c.a > 0.001f)))
        {
            foreach (var vertex in r.Mesh.vertices)
            {
                var world = r.transform.TransformPoint(vertex);
                if (first) { bounds = new Bounds(world, Vector3.zero); first = false; }
                else bounds.Encapsulate(world);
            }
        }
        var camera = new GameObject("LabCamera").AddComponent<Camera>();
        camera.orthographic = true;
        camera.transform.position = new Vector3(bounds.center.x, bounds.center.y, -10);
        camera.orthographicSize = Mathf.Max(bounds.extents.y, bounds.extents.x) * 1.08f;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.18f, 0.2f, 0.24f, 1);
        camera.nearClipPlane = 0.01f; camera.farClipPlane = 100;
        var target = new RenderTexture(1024, 1024, 24, RenderTextureFormat.ARGB32);
        target.Create(); camera.targetTexture = target;
        var image = new Texture2D(1024, 1024, TextureFormat.RGB24, false);
        var gesture = MakeGesture(model);
        bool neutral = Environment.GetEnvironmentVariable("ACTION_LAB_MODE") == "neutral";
        bool bodyProbe = Environment.GetEnvironmentVariable("ACTION_LAB_MODE") == "body-probe";
        bool bodyTurn = Environment.GetEnvironmentVariable("ACTION_LAB_MODE") == "body-turn";
        bool coordinated = Environment.GetEnvironmentVariable("ACTION_LAB_MODE") == "coordinated";
        string[] probeIds = { "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ", "ParamBodyAngleX2", "ParamBodyAngleY2", "ParamBodyAngleZ2" };
        var physics = model.GetComponent<CubismPhysicsController>();
        if (coordinated)
        {
            var modelJson = CubismModel3Json.LoadAtPath(modelPath, Load);
            var rig = CubismPhysics3Json.LoadFrom(modelJson.Physics3Json).ToRig();
            int excluded = 0;
            foreach (var sub in rig.SubRigs)
                for (int i = 0; i < sub.Output.Length; i++)
                    if (sub.Output[i].DestinationId == "ParamBodyAngleX2")
                    { sub.Output[i].Weight = 0; excluded++; }
            if (excluded != 1) throw new Exception("Expected exactly one X2 physics output, got " + excluded);
            physics.Initialize(rig);
        }
        CubismPhysics.UseFixedDeltaTime = true;
        Time.fixedDeltaTime = 1f / 30;
        physics.enabled = !neutral && !bodyProbe && !bodyTurn;
        // Prime SDK input indices before Stabilization resolves Source references.
        // In local 5-r.4, Evaluate sets SourceIndex only when Source is null;
        // Stabilization resolves Source without setting that index.
        if (physics.enabled) { physics.OnLateUpdate(); physics.Stabilization(); }
        for (int frame = 0; frame <= (bodyProbe ? 12 : neutral ? 2 : 120); frame++)
        {
            string probeLabel = "neutral";
            if (bodyProbe)
            {
                foreach (var p in model.Parameters) p.Value = p.DefaultValue;
                if (frame > 0)
                {
                    var p = model.Parameters.Single(p => p.Id == probeIds[(frame - 1) / 2]);
                    p.Value = frame % 2 == 1 ? p.MinimumValue * 0.8f : p.MaximumValue * 0.8f;
                    probeLabel = p.Id + "=" + p.Value;
                }
                File.WriteAllText(Path.Combine(output, frame.ToString("D4") + ".label.txt"), probeLabel);
            }
            else if (!neutral) gesture.SampleAnimation(model.gameObject, frame / 30f);
            var requested = Parameters(model);
            if (physics.enabled && frame > 0) physics.OnLateUpdate();
            if (frame == 39) UnityEngine.Debug.Log("BODY_DIAGNOSTIC physicsEnabled=" + physics.enabled + " fixed=" + CubismPhysics.UseFixedDeltaTime + " dt=" + Time.fixedDeltaTime + " values=" + string.Join(",", model.Parameters.Where(p => p.Id.Contains("Body")).Select(p => p.Id + "=" + p.Value)));
            model.ForceUpdateNow(); render.OnLateUpdate();
            if (masks != null) masks.OnLateUpdate();
            var watch = Stopwatch.StartNew();
            camera.Render(); RenderTexture.active = target;
            image.ReadPixels(new Rect(0, 0, 1024, 1024), 0, 0); image.Apply();
            watch.Stop();
            var state = new FrameState {
                frame = frame, actionTime = frame / 30.0, wallUtc = DateTime.UtcNow.ToString("O"),
                renderReadbackMs = watch.Elapsed.TotalMilliseconds,
                physics = !physics.enabled ? "disabled" : "SDK CubismPhysicsController; stabilized; dt=1/30; once per interval" + (coordinated ? "; X2 output weight=0; X2 owned by AnimationClip" : ""),
                writers = new[] { bodyProbe ? "isolated body pose" : neutral ? "defaults held" : physics.enabled ? "Unity AnimationClip.SampleAnimation + SDK physics fixed step" : "Unity AnimationClip.SampleAnimation; physics off", "manual CubismModel.ForceUpdateNow", "manual render/mask update; automatic writers disabled at initialization" }.Concat(writers).ToArray(),
                requested = requested, applied = Parameters(model),
                parts = model.Parts.Select(p => new PartState { id = p.Id, opacity = p.Opacity }).ToArray(),
                drawables = model.Drawables.Select((d, i) => new DrawableState {
                    id = d.Id, vertices = render.Renderers[i].Mesh.vertices, indices = d.Indices,
                    opacity = render.Renderers[i].Mesh.colors.Length == 0 ? 0 : render.Renderers[i].Mesh.colors[0].a,
                    sortingOrder = render.Renderers[i].MeshRenderer.sortingOrder,
                    enabled = render.Renderers[i].MeshRenderer.enabled
                }).ToArray()
            };
            File.WriteAllBytes(Path.Combine(output, frame.ToString("D4") + ".png"), image.EncodeToPNG());
            File.WriteAllText(Path.Combine(output, frame.ToString("D4") + ".json"), JsonUtility.ToJson(state));
        }
        RenderTexture.active = null;
        camera.targetTexture = null;
        UnityEngine.Object.DestroyImmediate(image); target.Release();
        File.WriteAllText(Path.Combine(output, "complete.txt"), coordinated ? "head leads; X2 follows; SDK physics excluding X2; not user approved\n" : bodyProbe ? "independent body poses; not a motion sequence; physics off\n" : bodyTurn ? "isolated X2 curve; physics off; not user approved\n" : neutral ? "neutral; physics off\n" : "turn-nod-return candidate; SDK physics fixed step; not user approved\n");
    }
}
