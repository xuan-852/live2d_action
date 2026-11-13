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
        public float[] goal, goalScreen, actualScreen;
        public float[] nativeNodeScreen, patchNodeScreen;
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
        var marginText=Environment.GetEnvironmentVariable("ACTION_LAB_CAMERA_MARGIN");
        float margin=string.IsNullOrEmpty(marginText)?1.08f:float.Parse(marginText,System.Globalization.CultureInfo.InvariantCulture);
        if(float.IsNaN(margin)||float.IsInfinity(margin)||margin<1.08f||margin>2f)throw new ArgumentException("Lab camera margin");
        scene.Camera.orthographicSize = Mathf.Max(bounds.extents.y, bounds.extents.x) * margin;
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
        if(frame.wholeArms!=null)
        {
            float[][] chains={frame.wholeArms.nativeChain,frame.wholeArms.patchChain};
            float[][] screens=new float[2][];
            for(int side=0;side<2;side++)
            {
                screens[side]=new float[chains[side].Length];
                for(int j=0;j<chains[side].Length;j+=2)
                {
                    var v=scene.Camera.WorldToScreenPoint(scene.Model.transform.TransformPoint(new Vector3(chains[side][j],chains[side][j+1],0)));
                    screens[side][j]=v.x;screens[side][j+1]=1024-v.y;
                }
            }
            frame.nativeNodeScreen=screens[0];frame.patchNodeScreen=screens[1];
        }
        if(frame.goal!=null && frame.wholeArms!=null)
        {
            frame.actualScreen=new float[4];
            for(int j=0;j<2;j++)
            {
                var chain=frame.wholeArms.nativeChain;
                var p=scene.Camera.WorldToScreenPoint(scene.Model.transform.TransformPoint(new Vector3(chain[2+j*2],chain[3+j*2],0)));
                frame.actualScreen[j*2]=p.x;frame.actualScreen[j*2+1]=1024-p.y;
            }
        }
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
        var physicsOverride=Environment.GetEnvironmentVariable("ACTION_LAB_PHYSICS_OVERRIDE");
        if (excludeX2 || !string.IsNullOrEmpty(physicsOverride))
        {
            var modelJson = CubismModel3Json.LoadAtPath(modelPath, Load);
            var rig = CubismPhysics3Json.LoadFrom(string.IsNullOrEmpty(physicsOverride)?modelJson.Physics3Json:File.ReadAllText(physicsOverride)).ToRig();
            var excluded = 0;
            foreach (var sub in rig.SubRigs)
                for (var i = 0; i < sub.Output.Length; i++)
                    if (excludeX2 && sub.Output[i].DestinationId == "ParamBodyAngleX2") { sub.Output[i].Weight = 0; excluded++; }
            if (excludeX2 && excluded != 1) throw new Exception("Expected exactly one X2 physics output, got " + excluded);
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

    [Serializable] public class ControlCommand { public float[] q, body, goal, head; public WholeArmSession.Pose patch; }

    [Serializable] public class StatefulCommand
    {
        public bool abort;
        public float[] q, body, headBody, nativeFingers, gaze, face;
        public WholeArmSession.Pose patch;
    }

    static string ReadStatefulCommand(string path)
    {
        // An atomically published file can still briefly retain a Windows rename
        // handle. Permit delete sharing and retry only sharing/lock violations;
        // invalid content and unrelated IO errors must remain visible failures.
        var watch=Stopwatch.StartNew();
        while(true)
        {
            try
            {
                using(var stream=new FileStream(path,FileMode.Open,FileAccess.Read,FileShare.ReadWrite|FileShare.Delete))
                using(var reader=new StreamReader(stream)) return reader.ReadToEnd();
            }
            catch(IOException e)
            {
                int code=e.HResult&0xffff;
                if((code!=32&&code!=33)||watch.ElapsedMilliseconds>=2000) throw;
                System.Threading.Thread.Sleep(5);
            }
        }
    }

    // Separate opt-in protocol: never changes previous experimental entry points.
    public static void RunStatefulControl()
    {
        try
        {
            var output=Required("ACTION_LAB_OUTPUT"); Guard(output);
            var scene=BuildScene(Required("ACTION_LAB_MODEL"),output);
            bool physics=Required("ACTION_LAB_STATEFUL_PHYSICS")=="on";
            ConfigurePhysics(scene,Required("ACTION_LAB_MODEL"),physics,false);
            int count=int.Parse(Required("ACTION_LAB_CONTROL_FRAMES"));
            var bridge=Required("ACTION_LAB_CONTROL_BRIDGE");
            bool fiveAxes=Environment.GetEnvironmentVariable("ACTION_LAB_STATEFUL_NATIVE_AXES")=="5";
            string[] ids=fiveAxes?new[]{"Param94","Param95","Param97","Param98","Param99","ParamBodyAngleX","ParamBodyAngleY"}:
                new[]{"Param94","Param97","Param99","ParamBodyAngleX","ParamBodyAngleY"};
            int nativeAxes=fiveAxes?5:3;
            bool fingerControl=Environment.GetEnvironmentVariable("ACTION_LAB_NATIVE_FINGERS")=="on";
            string[] fingerIds={"Param102","Param103","Param105","Param106","Param107"};
            var previousFingers=new float[5];var previousFingerSteps=new float[5];
            bool gazeControl=Environment.GetEnvironmentVariable("ACTION_LAB_GAZE")=="on";
            string[] gazeIds={"ParamEyeBallX","ParamEyeBallY"};
            var previousGaze=new float[2];var previousGazeSteps=new float[2];
            var previousFace=new[]{1f,1f,0f,0f,0f,0f};var previousFaceSteps=new float[6];
            var speedText=Environment.GetEnvironmentVariable("ACTION_LAB_STATEFUL_PARAMETER_SPEED");
            float parameterSpeed=string.IsNullOrEmpty(speedText)?45f:
                float.Parse(speedText,System.Globalization.CultureInfo.InvariantCulture);
            if(float.IsNaN(parameterSpeed)||float.IsInfinity(parameterSpeed)||parameterSpeed<=0||parameterSpeed>(fiveAxes?2700:180))
                throw new ArgumentException("Stateful speed budget");
            float frameSlew=parameterSpeed/30f;
            float[] previous=new float[nativeAxes];
            var timeoutText=Environment.GetEnvironmentVariable("ACTION_LAB_COMMAND_TIMEOUT");
            double commandTimeout=string.IsNullOrEmpty(timeoutText)?60:double.Parse(timeoutText,System.Globalization.CultureInfo.InvariantCulture);
            if(double.IsNaN(commandTimeout)||double.IsInfinity(commandTimeout)||commandTimeout<60||commandTimeout>1800)throw new ArgumentException("Lab command timeout");
            for(int i=0;i<count;i++)
            {
                string path=Path.Combine(bridge,i.ToString("D4")+".command.json");
                var wait=Stopwatch.StartNew();
                while(!File.Exists(path))
                {
                    if(wait.Elapsed.TotalSeconds>commandTimeout) throw new TimeoutException("Stateful command "+i);
                    System.Threading.Thread.Sleep(5);
                }
                var cmd=JsonUtility.FromJson<StatefulCommand>(ReadStatefulCommand(path));
                if(cmd.abort)
                {
                    File.WriteAllText(Path.Combine(output,"aborted.json"),"{\"nextFrame\":"+i+",\"reason\":\"provider failed\"}");
                    Finish(scene,output,"aborted provider; partial capture only");EditorApplication.Exit(1);return;
                }
                if(cmd.q==null || cmd.q.Length!=nativeAxes || cmd.body==null || cmd.body.Length!=2 || cmd.patch==null)
                    throw new ArgumentException("Stateful dimensions");
                float[] values=cmd.q.Concat(cmd.body).ToArray();
                for(int j=0;j<ids.Length;j++)
                {
                    var p=scene.Model.Parameters.Single(v=>v.Id==ids[j]); float v=values[j];
                    if(float.IsNaN(v)||float.IsInfinity(v)||v<p.MinimumValue||v>p.MaximumValue)
                        throw new ArgumentException("Stateful parameter bounds");
                    if(j<nativeAxes && Mathf.Abs(v-previous[j])>frameSlew+.0001f) throw new ArgumentException("Stateful frame slew");
                    p.Value=v;
                }
                previous=cmd.q;
                if(cmd.headBody!=null)
                {
                    if(cmd.headBody.Length!=6) throw new ArgumentException("Stateful head/body dimensions");
                    string[] contextIds={"ParamAngleX","ParamAngleY","ParamAngleZ","ParamBodyAngleX","ParamBodyAngleY","ParamBodyAngleZ"};
                    for(int j=0;j<contextIds.Length;j++)
                    {
                        var p=scene.Model.Parameters.Single(v=>v.Id==contextIds[j]); float v=cmd.headBody[j];
                        if(float.IsNaN(v)||float.IsInfinity(v)||v<p.MinimumValue||v>p.MaximumValue)
                            throw new ArgumentException("Stateful head/body bounds");
                        p.Value=v;
                    }
                }
                if(fingerControl)
                {
                    if(cmd.nativeFingers==null||cmd.nativeFingers.Length!=5)throw new ArgumentException("Native finger dimensions");
                    for(int j=0;j<5;j++)
                    {
                        float v=cmd.nativeFingers[j],step=v-previousFingers[j];
                        if(float.IsNaN(v)||float.IsInfinity(v)||v< -20.0001f||v>20.0001f||Mathf.Abs(step)>3.0001f||Mathf.Abs(step-previousFingerSteps[j])>.6001f)
                            throw new ArgumentException("Native finger range/frame speed/acceleration");
                        scene.Model.Parameters.Single(p=>p.Id==fingerIds[j]).Value=v;
                        previousFingers[j]=v;previousFingerSteps[j]=step;
                    }
                }
                else if(cmd.nativeFingers!=null)throw new ArgumentException("Native finger protocol not enabled");
                if(gazeControl)
                {
                    if(cmd.gaze==null||cmd.gaze.Length!=2)throw new ArgumentException("Gaze dimensions");
                    for(int j=0;j<2;j++)
                    {
                        float v=cmd.gaze[j],step=v-previousGaze[j];
                        if(float.IsNaN(v)||float.IsInfinity(v)||Mathf.Abs(v)>.8001f||Mathf.Abs(step)>.1001f||Mathf.Abs(step-previousGazeSteps[j])>.0168f)
                            throw new ArgumentException("Gaze range/frame speed/acceleration");
                        scene.Model.Parameters.Single(p=>p.Id==gazeIds[j]).Value=v;
                        previousGaze[j]=v;previousGazeSteps[j]=step;
                    }
                }
                else if(cmd.gaze!=null)throw new ArgumentException("Gaze protocol not enabled");
                if(Environment.GetEnvironmentVariable("ACTION_LAB_FACE")=="on")
                {
                    string[] faceIds={"ParamEyeLOpen","ParamEyeROpen","ParamMouthOpenY","ParamMouthForm","ParamEyeLSmile","ParamEyeRSmile"};
                    if(cmd.face==null||cmd.face.Length!=faceIds.Length)throw new ArgumentException("Face dimensions");
                    for(int j=0;j<faceIds.Length;j++)
                    {
                        var p=scene.Model.Parameters.Single(v=>v.Id==faceIds[j]);float v=cmd.face[j];float step=v-previousFace[j];
                        if(float.IsNaN(v)||float.IsInfinity(v)||v<p.MinimumValue||v>p.MaximumValue)throw new ArgumentException("Face bounds");
                        if(Mathf.Abs(step)>.1001f||Mathf.Abs(step-previousFaceSteps[j])>.0201f)throw new ArgumentException("Face frame speed/acceleration");
                        p.Value=v;
                        previousFace[j]=v;previousFaceSteps[j]=step;
                    }
                }
                else if(cmd.face!=null)throw new ArgumentException("Face protocol not enabled");
                scene.Whole.ControlPose=cmd.patch;
                var requested=Snapshot(scene.Model);
                if(physics && i>0) scene.Physics.OnLateUpdate();
                scene.Model.ForceUpdateNow(); scene.Render.OnLateUpdate();
                if(scene.Masks!=null) scene.Masks.OnLateUpdate();
                WriteFrame(scene,output,new Frame { frame=i,actionTime=i/30.0,
                    physicsState=physics?"SDK fixed dt=1/30, original rig":"disabled",
                    requested=requested,applied=Snapshot(scene.Model),
                    writers=new[]{"stateful arm command","SDK/Core","final mesh readback"},
                    parts=scene.Model.Parts.Select(p=>new PartState{id=p.Id,opacity=p.Opacity}).ToArray() },null);
                File.WriteAllText(Path.Combine(bridge,i.ToString("D4")+".ready"),"ok");
            }
            Finish(scene,output,"stateful arms; fixed action time; not realtime"); EditorApplication.Exit(0);
        }
        catch(Exception e) { UnityEngine.Debug.LogException(e); EditorApplication.Exit(1); }
    }

    public static void RunControl()
    {
        try
        {
            var output=Required("ACTION_LAB_OUTPUT"); Guard(output);
            var scene=BuildScene(Required("ACTION_LAB_MODEL"),output);
            ConfigurePhysics(scene,Required("ACTION_LAB_MODEL"),true,false);
            int count=int.Parse(Required("ACTION_LAB_CONTROL_FRAMES"));
            var bridge=Required("ACTION_LAB_CONTROL_BRIDGE");
            float[] previous={0,0};
            for(int i=0;i<count;i++)
            {
                string path=Path.Combine(bridge,i.ToString("D4")+".command.json");
                var wait=Stopwatch.StartNew();
                while(!File.Exists(path))
                {
                    if(wait.Elapsed.TotalSeconds>60) throw new TimeoutException("Control command "+i);
                    System.Threading.Thread.Sleep(5);
                }
                var cmd=JsonUtility.FromJson<ControlCommand>(File.ReadAllText(path));
                if(cmd.q.Length!=2 || cmd.body.Length!=2 || cmd.goal.Length!=4) throw new Exception("Control dimensions");
                string[] ids={"Param94","Param97","ParamBodyAngleX","ParamBodyAngleY"};
                float[] values={cmd.q[0],cmd.q[1],cmd.body[0],cmd.body[1]};
                for(int j=0;j<4;j++)
                {
                    var p=scene.Model.Parameters.Single(v=>v.Id==ids[j]);
                    float v=values[j];
                    if(float.IsNaN(v)||float.IsInfinity(v)||v<p.MinimumValue||v>p.MaximumValue) throw new Exception("Control bounds");
                    if(j<2 && Mathf.Abs(v-previous[j])>1.5001f) throw new Exception("Control slew");
                    p.Value=v;
                }
                previous=cmd.q;
                if(cmd.patch!=null)
                {
                    if(scene.Whole==null) throw new Exception("Patch requires whole session");
                    scene.Whole.ControlPose=cmd.patch;
                }
                if(cmd.head!=null)
                {
                    if(cmd.head.Length!=3) throw new Exception("Head dimensions");
                    string[] headIds={"ParamAngleX","ParamAngleY","ParamAngleZ"};
                    for(int j=0;j<3;j++)
                    {
                        float v=cmd.head[j];
                        if(float.IsNaN(v)||float.IsInfinity(v)||Mathf.Abs(v)>15) throw new Exception("Head bound");
                        scene.Model.Parameters.Single(p=>p.Id==headIds[j]).Value=v;
                    }
                }
                var requested=Snapshot(scene.Model);
                if(i>0) scene.Physics.OnLateUpdate();
                scene.Model.ForceUpdateNow(); scene.Render.OnLateUpdate();
                if(scene.Masks!=null) scene.Masks.OnLateUpdate();
                var frame=new Frame { frame=i, actionTime=i/30.0, goal=cmd.goal,
                    physicsState="sdk physics fixed dt=1/30; interactive feedback",
                    requested=requested, applied=Snapshot(scene.Model),
                    writers=new[]{"bounded feedback command", "sdk physics", "Core/render"},
                    parts=scene.Model.Parts.Select(p=>new PartState{id=p.Id,opacity=p.Opacity}).ToArray() };
                frame.goalScreen=new float[4];
                for(int j=0;j<2;j++)
                {
                    var v=scene.Camera.WorldToScreenPoint(scene.Model.transform.TransformPoint(new Vector3(cmd.goal[j*2],cmd.goal[j*2+1],0)));
                    frame.goalScreen[j*2]=v.x; frame.goalScreen[j*2+1]=1024-v.y;
                }
                WriteFrame(scene,output,frame,null);
                File.WriteAllText(Path.Combine(bridge,i.ToString("D4")+".ready"),"ok");
            }
            Finish(scene,output,"interactive control; fixed action time, not realtime"); EditorApplication.Exit(0);
        }
        catch(Exception e) { UnityEngine.Debug.LogException(e); EditorApplication.Exit(1); }
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
