using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Live2D.Cubism.Core;
using Live2D.Cubism.Framework.Json;
using Live2D.Cubism.Framework.Physics;
using Live2D.Cubism.Rendering;
using Live2D.Cubism.Rendering.Masking;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Batch entry points for pose calibration and curve-driven capture.
// Complements LabCapture.cs; does not modify it. No automatic parameter writers.
public static class LabMotion
{
    [Serializable] public class PoseSpec { public string label; public string[] ids; public float[] values; }
    [Serializable] public class PoseFile { public PoseSpec[] poses; public bool physics; }
    [Serializable] public class CurveChannel { public float[] values; }
    [Serializable] public class CurveFile
    {
        public float fps = 30; public float duration = 4; public bool physics; public bool excludeX2;
        public string[] ids; public CurveChannel[] channels;
    }
    [Serializable] public class ParameterState { public string id; public float value, minimum, maximum, defaultValue; }
    [Serializable] public class PartState { public string id; public float opacity; }
    [Serializable] public class DrawableGeometry
    {
        public string id; public float cx, cy, minx, miny, maxx, maxy; public int vertices;
    }
    [Serializable] public class Frame
    {
        public int frame; public double actionTime; public string physicsState;
        public string label; public double pairingMaxDelta;
        public string[] writers; public ParameterState[] requested, applied;
        public PartState[] parts; public DrawableGeometry[] drawables;
        public double renderReadbackMs;
        public WholeArmSession.Readback wholeArms;
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

    static string Required(string name)
    {
        var value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(value)) throw new ArgumentException("Set " + name);
        return value;
    }

    static ParameterState[] Snapshot(CubismModel model) => model.Parameters.Select(p => new ParameterState
    {
        id = p.Id, value = p.Value, minimum = p.MinimumValue, maximum = p.MaximumValue, defaultValue = p.DefaultValue
    }).ToArray();

    static CubismModel LoadModel(string modelPath, string savePngPath)
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var model = CubismModel3Json.LoadAtPath(modelPath, Load).ToModel();
        if (model == null) throw new Exception("Model creation failed");
        foreach (var behaviour in model.GetComponentsInChildren<Behaviour>()) behaviour.enabled = false;
        var render = model.GetComponent<CubismRenderController>();
        render.enabled = true;
        render.TryInitializeRenderers();
        var masks = model.GetComponent<CubismMaskController>();
        if (masks != null) masks.enabled = true;
        foreach (var p in model.Parameters) p.Value = p.DefaultValue;
        model.ForceUpdateNow();
        render.OnLateUpdate();
        if (masks != null) masks.OnLateUpdate();
        return model;
    }

    sealed class Scene
    {
        public CubismModel Model;
        public CubismRenderController Render;
        public CubismMaskController Masks;
        public Camera Camera;
        public RenderTexture Target;
        public Texture2D Image;
        public CubismPhysicsController Physics;
        public WholeArmSession Whole;
    }

    static Scene BuildScene(string modelPath, string output)
    {
        var scene = new Scene();
        scene.Model = LoadModel(modelPath, output);
        scene.Render = scene.Model.GetComponent<CubismRenderController>();
        scene.Masks = scene.Model.GetComponent<CubismMaskController>();
        var bounds = new Bounds();
        var first = true;
        foreach (var r in scene.Render.Renderers.Where(r => r.Mesh.colors.Any(c => c.a > 0.001f)))
        {
            foreach (var vertex in r.Mesh.vertices)
            {
                var world = r.transform.TransformPoint(vertex);
                if (first) { bounds = new Bounds(world, Vector3.zero); first = false; }
                else bounds.Encapsulate(world);
            }
        }
        scene.Camera = new GameObject("LabCamera").AddComponent<Camera>();
        scene.Camera.orthographic = true;
        scene.Camera.transform.position = new Vector3(bounds.center.x, bounds.center.y, -10);
        scene.Camera.orthographicSize = Mathf.Max(bounds.extents.y, bounds.extents.x) * 1.08f;
        scene.Camera.clearFlags = CameraClearFlags.SolidColor;
        scene.Camera.backgroundColor = new Color(0.18f, 0.2f, 0.24f, 1);
        scene.Camera.nearClipPlane = 0.01f;
        scene.Camera.farClipPlane = 100;
        scene.Target = new RenderTexture(1024, 1024, 24, RenderTextureFormat.ARGB32);
        scene.Target.Create();
        scene.Camera.targetTexture = scene.Target;
        scene.Image = new Texture2D(1024, 1024, TextureFormat.RGB24, false);
        scene.Physics = scene.Model.GetComponent<CubismPhysicsController>();
        var patchPath = Environment.GetEnvironmentVariable("ACTION_LAB_PATCH");
        if (!string.IsNullOrEmpty(patchPath)) scene.Whole = new WholeArmSession(scene.Model, patchPath);
        return scene;
    }

    static DrawableGeometry[] Geometry(Scene scene)
    {
        var result = new DrawableGeometry[scene.Model.Drawables.Length];
        for (int i = 0; i < result.Length; i++)
        {
            var d = scene.Model.Drawables[i];
            // Whole mode records the postprocessed mesh actually submitted to render,
            // not the unmodified Core vertices of the patched arm.
            var verts = scene.Whole == null ? d.VertexPositions : scene.Render.Renderers[i].Mesh.vertices;
            var g = new DrawableGeometry { id = d.Id, vertices = verts.Length, minx = float.MaxValue, miny = float.MaxValue, maxx = float.MinValue, maxy = float.MinValue };
            var sum = Vector3.zero;
            foreach (var v in verts)
            {
                sum += v;
                g.minx = Mathf.Min(g.minx, v.x); g.maxx = Mathf.Max(g.maxx, v.x);
                g.miny = Mathf.Min(g.miny, v.y); g.maxy = Mathf.Max(g.maxy, v.y);
            }
            if (verts.Length > 0) { g.cx = sum.x / verts.Length; g.cy = sum.y / verts.Length; }
            result[i] = g;
        }
        return result;
    }

    /// <summary>
    /// CubismRenderer double-buffers meshes: after a model update the MeshFilter still shows
    /// the previous update's buffer. Swap once more so the rendered image matches the state
    /// that was just written, and report the remaining disagreement with the unmanaged model.
    /// </summary>
    static double PresentCurrent(Scene scene)
    {
        foreach (var renderer in scene.Render.Renderers) renderer.SwapMeshes();
        var maxDelta = 0.0;
        for (var i = 0; i < scene.Model.Drawables.Length; i++)
        {
            var mesh = scene.Render.Renderers[i].Mesh.vertices;
            var truth = scene.Model.Drawables[i].VertexPositions;
            if (mesh.Length != truth.Length) return double.PositiveInfinity;
            for (var v = 0; v < mesh.Length; v++) maxDelta = Math.Max(maxDelta, (mesh[v] - truth[v]).magnitude);
        }
        return maxDelta;
    }

    static void WriteFrame(Scene scene, string output, Frame frame, string label)
    {
        frame.pairingMaxDelta = PresentCurrent(scene);
        if (scene.Whole != null) { scene.Whole.Apply(frame.frame); frame.wholeArms=scene.Whole.Current; }
        var watch = Stopwatch.StartNew();
        scene.Camera.Render();
        RenderTexture.active = scene.Target;
        scene.Image.ReadPixels(new Rect(0, 0, 1024, 1024), 0, 0);
        scene.Image.Apply();
        watch.Stop();
        frame.renderReadbackMs = watch.Elapsed.TotalMilliseconds;
        frame.drawables = Geometry(scene);
        if (label != null) frame.label = label;
        File.WriteAllBytes(Path.Combine(output, frame.frame.ToString("D4") + ".png"), scene.Image.EncodeToPNG());
        var diagnosticFrame = Environment.GetEnvironmentVariable("ACTION_LAB_LAYER_FRAME");
        if (diagnosticFrame == frame.frame.ToString()) LayerDiagnostic(scene, output);
        if(scene.Whole!=null) scene.Whole.Restore();
        File.WriteAllText(Path.Combine(output, frame.frame.ToString("D4") + ".json"), JsonUtility.ToJson(frame));
    }

    [Serializable] public class LayerState
    {
        public string id, shader; public int order, queue; public float z;
        public string[] masks;
        public Vector3[] vertices;
        public Vector3[] submittedVertices;
        public int meshId, submittedMeshId;
    }
    [Serializable] public class LayerStates { public LayerState[] drawables; }
    static void LayerDiagnostic(Scene scene, string output)
    {
        var folder=Path.Combine(output,"layers"); Directory.CreateDirectory(folder);
        var renderers=scene.Render.Renderers;
        var enabled=renderers.Select(r=>r.MeshRenderer.enabled).ToArray();
        var orders=renderers.Select(r=>r.MeshRenderer.sortingOrder).ToArray();
        var positions=renderers.Select(r=>r.transform.localPosition).ToArray();
        var states=scene.Model.Drawables.Select(d=>{
            var r=d.GetComponent<CubismRenderer>(); var m=r.MeshRenderer.sharedMaterial;
            return new LayerState { id=d.Id, order=r.MeshRenderer.sortingOrder,
                z=r.transform.localPosition.z, shader=m.shader.name, queue=m.renderQueue,
                masks=d.Masks.Select(v=>v.Id).ToArray(), vertices=r.Mesh.vertices,
                submittedVertices=r.GetComponent<MeshFilter>().sharedMesh.vertices,
                meshId=r.Mesh.GetInstanceID(),submittedMeshId=r.GetComponent<MeshFilter>().sharedMesh.GetInstanceID() };
        }).ToArray();
        File.WriteAllText(Path.Combine(folder,"states.json"),JsonUtility.ToJson(new LayerStates {drawables=states},true));
        Action<string> capture=name=>{
            scene.Camera.Render(); RenderTexture.active=scene.Target;
            scene.Image.ReadPixels(new Rect(0,0,1024,1024),0,0); scene.Image.Apply();
            File.WriteAllBytes(Path.Combine(folder,name+".png"),scene.Image.EncodeToPNG());
        };
        try
        {
            capture("baseline");
            var ids=ActionLab.Recovered.SandboxArmGeometry.TargetIds;
            for(int i=0;i<renderers.Length;i++)
                if(ids.Contains(scene.Model.Drawables[i].Id)) renderers[i].MeshRenderer.sortingOrder=2000+i;
            capture("all-patch-front");
            for(int i=0;i<renderers.Length;i++) renderers[i].MeshRenderer.sortingOrder=orders[i];
            var upper=scene.Model.Drawables.Single(d=>d.Id=="ArtMesh263").GetComponent<CubismRenderer>();
            var upperOrder=upper.MeshRenderer.sortingOrder;
            upper.MeshRenderer.sortingOrder=1500;
            capture("upper-skin-front");
            upper.MeshRenderer.sortingOrder=upperOrder;
            var sleeve=scene.Model.Drawables.Single(d=>d.Id=="ArtMesh252").GetComponent<CubismRenderer>();
            var cap=scene.Model.Drawables.Single(d=>d.Id=="ArtMesh250").GetComponent<CubismRenderer>();
            var upperPosition=upper.transform.localPosition;
            upper.MeshRenderer.sortingOrder=sleeve.MeshRenderer.sortingOrder;
            var relative=upperPosition;relative.z=(sleeve.transform.localPosition.z+cap.transform.localPosition.z)*.5f;
            upper.transform.localPosition=relative;
            capture("upper-between-cap-sleeve");
            upper.transform.localPosition=upperPosition;upper.MeshRenderer.sortingOrder=upperOrder;
            foreach(var id in new[]{"ArtMesh250","ArtMesh251","ArtMesh252","ArtMesh262"})
            {
                var r=scene.Model.Drawables.Single(d=>d.Id==id).GetComponent<CubismRenderer>();
                var was=r.MeshRenderer.enabled;r.MeshRenderer.enabled=false;
                capture("without-"+id);r.MeshRenderer.enabled=was;
            }
            var nativeIds=new[]{"ArtMesh224","ArtMesh225","ArtMesh226","ArtMesh227","ArtMesh230",
                "ArtMesh231","ArtMesh232","ArtMesh233","ArtMesh234","ArtMesh235","ArtMesh236",
                "ArtMesh238","ArtMesh239","ArtMesh240","ArtMesh241","ArtMesh242","ArtMesh243",
                "ArtMesh244","ArtMesh245","ArtMesh246","ArtMesh247","ArtMesh248"};
            var nativeDistal=new[]{"ArtMesh231","ArtMesh232","ArtMesh233","ArtMesh234","ArtMesh235","ArtMesh236","ArtMesh238"};
            for(int i=0;i<renderers.Length;i++)
                if(nativeDistal.Contains(scene.Model.Drawables[i].Id)) renderers[i].MeshRenderer.sortingOrder=1000;
            capture("native-distal-front");
            for(int i=0;i<renderers.Length;i++) renderers[i].MeshRenderer.sortingOrder=orders[i];
            for(int i=0;i<renderers.Length;i++)
                if(nativeIds.Contains(scene.Model.Drawables[i].Id)) renderers[i].MeshRenderer.sortingOrder=1000;
            capture("native-group-front");
            for(int i=0;i<renderers.Length;i++)
            {
                renderers[i].MeshRenderer.sortingOrder=orders[i];
                renderers[i].MeshRenderer.enabled=enabled[i] && nativeIds.Contains(scene.Model.Drawables[i].Id);
            }
            capture("native-group-only");
            var inspect=ids.Concat(nativeIds).Concat(ActionLab.Recovered.SandboxArmGeometry.ClothIds).Distinct();
            foreach(var id in inspect)
            {
                for(int i=0;i<renderers.Length;i++) renderers[i].MeshRenderer.enabled=scene.Model.Drawables[i].Id==id;
                capture(id);
            }
        }
        finally
        {
            for(int i=0;i<renderers.Length;i++)
            {
                renderers[i].MeshRenderer.enabled=enabled[i]; renderers[i].MeshRenderer.sortingOrder=orders[i];
                renderers[i].transform.localPosition=positions[i];
            }
            // Restore the image paired with the ordinary frame, not the last diagnostic.
            capture("restored");
        }
    }

    static void ConfigurePhysics(Scene scene, string modelPath, bool enabled, bool excludeX2)
    {
        CubismPhysics.UseFixedDeltaTime = true;
        Time.fixedDeltaTime = 1f / 30;
        scene.Physics.enabled = enabled;
        if (!enabled) return;
        if (excludeX2)
        {
            var modelJson = CubismModel3Json.LoadAtPath(modelPath, Load);
            var rig = CubismPhysics3Json.LoadFrom(modelJson.Physics3Json).ToRig();
            var excluded = 0;
            foreach (var sub in rig.SubRigs)
                for (var i = 0; i < sub.Output.Length; i++)
                    if (sub.Output[i].DestinationId == "ParamBodyAngleX2") { sub.Output[i].Weight = 0; excluded++; }
            if (excluded != 1) throw new Exception("Expected exactly one X2 physics output, got " + excluded);
            scene.Physics.Initialize(rig);
        }
        scene.Physics.OnLateUpdate();
        scene.Physics.Stabilization();
    }

    static void Finish(Scene scene, string output, string note)
    {
        RenderTexture.active = null;
        if(scene.Whole!=null) scene.Whole.Restore();
        scene.Camera.targetTexture = null;
        UnityEngine.Object.DestroyImmediate(scene.Image);
        scene.Target.Release();
        File.WriteAllText(Path.Combine(output, "complete.txt"), note + "\n");
    }

    static void Guard(string output)
    {
        if (Directory.Exists(output)) throw new IOException("Refusing to overwrite run: " + output);
        Directory.CreateDirectory(output);
    }

    public static void RunPoseLab()
    {
        try
        {
            var modelPath = Required("ACTION_LAB_MODEL");
            var output = Required("ACTION_LAB_OUTPUT");
            var poses = JsonUtility.FromJson<PoseFile>(File.ReadAllText(Required("ACTION_LAB_POSES")));
            Guard(output);
            var scene = BuildScene(modelPath, output);
            ConfigurePhysics(scene, modelPath, poses.physics, false);
            for (var index = 0; index < poses.poses.Length; index++)
            {
                var pose = poses.poses[index];
                foreach (var p in scene.Model.Parameters) p.Value = p.DefaultValue;
                for (var i = 0; i < pose.ids.Length; i++)
                {
                    var parameter = scene.Model.Parameters.Single(p => p.Id == pose.ids[i]);
                    parameter.Value = pose.values[i];
                }
                if (scene.Physics.enabled) scene.Physics.OnLateUpdate();
                scene.Model.ForceUpdateNow();
                scene.Render.OnLateUpdate();
                if (scene.Masks != null) scene.Masks.OnLateUpdate();
                WriteFrame(scene, output, new Frame
                {
                    frame = index, actionTime = 0, physicsState = poses.physics ? "sdk physics one step" : "disabled",
                    writers = new[] { "pose-lab: defaults then explicit values" },
                    requested = Snapshot(scene.Model), applied = Snapshot(scene.Model),
                    parts = scene.Model.Parts.Select(p => new PartState { id = p.Id, opacity = p.Opacity }).ToArray()
                }, pose.label);
                File.WriteAllText(Path.Combine(output, index.ToString("D4") + ".label.txt"), pose.label);
            }
            Finish(scene, output, "pose lab; " + poses.poses.Length + " independent poses; not a motion sequence");
            EditorApplication.Exit(0);
        }
        catch (Exception e) { UnityEngine.Debug.LogException(e); EditorApplication.Exit(1); }
    }

    public static void RunMotion()
    {
        try
        {
            var modelPath = Required("ACTION_LAB_MODEL");
            var output = Required("ACTION_LAB_OUTPUT");
            var curve = JsonUtility.FromJson<CurveFile>(File.ReadAllText(Required("ACTION_LAB_CURVES")));
            Guard(output);
            var scene = BuildScene(modelPath, output);
            var clip = new AnimationClip { name = "batch" };
            for (var i = 0; i < curve.ids.Length; i++)
            {
                var parameter = scene.Model.Parameters.Single(p => p.Id == curve.ids[i]);
                var values = curve.channels[i].values;
                var keys = new Keyframe[values.Length];
                for (var k = 0; k < values.Length; k++) keys[k] = new Keyframe(k / curve.fps, values[k]);
                var animationCurve = new AnimationCurve(keys);
                for (var k = 0; k < values.Length; k++)
                {
                    AnimationUtility.SetKeyLeftTangentMode(animationCurve, k, AnimationUtility.TangentMode.Linear);
                    AnimationUtility.SetKeyRightTangentMode(animationCurve, k, AnimationUtility.TangentMode.Linear);
                }
                clip.SetCurve(AnimationUtility.CalculateTransformPath(parameter.transform, scene.Model.transform),
                    typeof(CubismParameter), "Value", animationCurve);
            }
            ConfigurePhysics(scene, modelPath, curve.physics, curve.excludeX2);
            var frames = Mathf.RoundToInt(curve.duration * curve.fps);
            for (var frame = 0; frame <= frames; frame++)
            {
                var time = frame / curve.fps;
                clip.SampleAnimation(scene.Model.gameObject, time);
                var requested = Snapshot(scene.Model);
                if (scene.Physics.enabled && frame > 0) scene.Physics.OnLateUpdate();
                scene.Model.ForceUpdateNow();
                scene.Render.OnLateUpdate();
                if (scene.Masks != null) scene.Masks.OnLateUpdate();
                WriteFrame(scene, output, new Frame
                {
                    frame = frame, actionTime = time,
                    physicsState = curve.physics ? "sdk physics; stabilized; dt=1/30" + (curve.excludeX2 ? "; X2 output weight=0" : "") : "disabled",
                    writers = new[] { "AnimationClip.SampleAnimation", curve.physics ? "sdk physics fixed step" : "physics off",
                        "manual ForceUpdateNow", "manual render/mask update" },
                    requested = requested, applied = Snapshot(scene.Model),
                    parts = scene.Model.Parts.Select(p => new PartState { id = p.Id, opacity = p.Opacity }).ToArray()
                }, null);
            }
            Finish(scene, output, "curve motion; " + (frames + 1) + " frames at " + curve.fps + "Hz; offline fixed step, not realtime performance");
            EditorApplication.Exit(0);
        }
        catch (Exception e) { UnityEngine.Debug.LogException(e); EditorApplication.Exit(1); }
    }
}
