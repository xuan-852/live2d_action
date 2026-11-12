using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Live2D.Cubism.Core;
using Live2D.Cubism.Rendering;
using ActionLab.Recovered;

// Explicit synchronous postprocess. No events, production lease, timers or legacy player.
public sealed class WholeArmSession
{
    [Serializable] public class Marker { public string id; public string mesh; public int[] indices; public float[] weights; }
    [Serializable] public class Pose
    {
        public float shoulder, elbow, wrist;
        public SecondaryArmProjection projection;
        public SecondaryArmGirdle girdle;
        public float forearmFront = 1, handFront = 1;
    }
    [Serializable] public class Input
    {
        public Pose[] poses; public Marker[] nativeMarkers;
        // legacy-z preserves the previous experiment. diagnostic-order is an
        // intentionally extreme front/back test, not a garment-layer solution.
        public string layerPolicy = "legacy-z";
        public bool rootedUpper;
        public bool nativeDistalFront;
        public float[] correctedPatchNodes;
    }
    [Serializable] public class Readback
    {
        public float[] nativeChain, patchChain, sourcePatchChain;
        public float[] areaGradient, patchActuators;
        public float[] meshAreaRatios, meshAreaGradients;
        public float minTriangleAreaRatio, maxPatchDisplacement, restoredMeshError;
        public int flippedTriangles, modifiedMeshes;
        public float forearmFront, handFront, upperProjection, foreProjection;
        public string layerPolicy, sdkSortingMode;
        public int[] actualSortingOrders;
        public string[] nativeDistalIds;
        public int[] nativeDistalSortingOrders;
        public Vector2[] attachmentControls;
        public string scope = "same-frame material markers, not anatomical truth; palm plane/fingers unknown";
    }
    sealed class Binding
    {
        public string id;
        public CubismDrawable drawable;
        public CubismRenderer renderer;
        public Vector3[] rest, source, final;
        public int[] triangles;
        public Vector3 position;
        public int originalOrder;
    }
    readonly CubismModel model;
    readonly Input input;
    readonly List<Binding> bindings = new List<Binding>();
    readonly ShoulderConnection connection;
    readonly Dictionary<CubismRenderer,int> distalOrders = new Dictionary<CubismRenderer,int>();
    public Readback Current;
    public Pose ControlPose;

    public WholeArmSession(CubismModel target, string path)
    {
        model = target;
        input = JsonUtility.FromJson<Input>(File.ReadAllText(path));
        if (input == null || input.poses == null || input.nativeMarkers == null)
            throw new ArgumentException("Missing whole-arm tracks/markers");
        foreach (var id in SandboxArmGeometry.TargetIds)
        {
            var d = model.Drawables.Single(v => v.Id == id);
            var r = d.GetComponent<CubismRenderer>();
            bindings.Add(new Binding { id=id, drawable=d, renderer=r, rest=d.VertexPositions,
                triangles=d.Indices, position=r.transform.localPosition, originalOrder=r.MeshRenderer.sortingOrder });
        }
        var skin = bindings.Single(b => b.id == "ArtMesh251");
        var fore = bindings.Single(b => b.id == "ArtMesh261");
        var hand = bindings.Single(b => b.id == "ArtMesh253");
        Vector2[] corrected=null;
        if(input.correctedPatchNodes!=null && input.correctedPatchNodes.Length>0)
        {
            if(input.correctedPatchNodes.Length!=8) throw new ArgumentException("Expected four corrected patch nodes");
            corrected=Enumerable.Range(0,4).Select(i=>new Vector2(input.correctedPatchNodes[2*i],input.correctedPatchNodes[2*i+1])).ToArray();
            if(corrected.Any(p=>!Finite(p.x)||!Finite(p.y))) throw new ArgumentException("Nonfinite patch nodes");
        }
        connection = new ShoulderConnection(skin.rest, skin.triangles, fore.rest, fore.triangles, hand.rest, hand.triangles,corrected);
        foreach(var d in model.Drawables.Where(d=>new[]{"ArtMesh231","ArtMesh232","ArtMesh233","ArtMesh234","ArtMesh235","ArtMesh236","ArtMesh238"}.Contains(d.Id)))
        { var r=d.GetComponent<CubismRenderer>(); distalOrders[r]=r.MeshRenderer.sortingOrder; }
    }
    static bool Finite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);
    static float Area(Vector3 a, Vector3 b, Vector3 c) => (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);

    // Local geometry derivative only; no mesh commit, Core/physics step or new
    // binding. The controller uses a margin; actual safety remains in readback.
    float[] MeasureAreas(Pose pose)
    {
        var areas=new List<float>();
        foreach(var b in bindings)
        {
            float minimum=1;
            var vertices=new Vector3[b.source.Length];
            for(int i=0;i<vertices.Length;i++)
            {
                Vector2 p=connection.Deform(b.source[i],b.rest[i],b.id,pose.shoulder,pose.elbow,pose.wrist,pose.projection,input.rootedUpper);
                vertices[i]=new Vector3(p.x,p.y,b.source[i].z);
            }
            for(int k=0;k<b.triangles.Length;k+=3)
            {
                int a=b.triangles[k],c=b.triangles[k+1],d=b.triangles[k+2];
                float before=Area(b.source[a],b.source[c],b.source[d]);
                if(Mathf.Abs(before)<1e-9f) continue;
                minimum=Mathf.Min(minimum,Area(vertices[a],vertices[c],vertices[d])/before);
            }
            areas.Add(minimum);
        }
        return areas.ToArray();
    }

    public void Apply(int frame)
    {
        if (frame >= input.poses.Length) throw new ArgumentException("Whole-arm sample count mismatch");
        var pose = ControlPose ?? input.poses[frame];
        if (!Finite(pose.shoulder) || !Finite(pose.elbow) || !Finite(pose.wrist) || !pose.projection.ReachValid || !pose.girdle.Valid)
            throw new ArgumentException("Invalid patch pose/projection");
        foreach (var b in bindings) b.source = b.drawable.VertexPositions;
        connection.Update(bindings.Single(b=>b.id=="ArtMesh251").source,
                          bindings.Single(b=>b.id=="ArtMesh261").source,
                          bindings.Single(b=>b.id=="ArtMesh253").source, pose.girdle);
        Current = new Readback { minTriangleAreaRatio=1, modifiedMeshes=bindings.Count };
        Current.patchActuators=new[]{pose.shoulder,pose.elbow,pose.wrist};
        Current.sourcePatchChain = connection.MaterialChain(id=>bindings.Single(b=>b.id==id).source)
            .SelectMany(p=>new[]{p.x,p.y}).ToArray();
        Current.forearmFront=pose.forearmFront; Current.handFront=pose.handFront;
        Current.upperProjection=pose.projection.upperLength; Current.foreProjection=pose.projection.forearmLength;
        Current.layerPolicy=input.layerPolicy;
        Current.sdkSortingMode=model.GetComponent<CubismRenderController>().SortingMode.ToString();
        // Calculate every mesh before committing. Failures are not half-applied.
        foreach (var b in bindings)
        {
            b.final = new Vector3[b.source.Length];
            for (int i=0; i<b.source.Length; i++)
            {
                Vector2 p = connection.Deform(b.source[i], b.rest[i], b.id, pose.shoulder, pose.elbow, pose.wrist, pose.projection, input.rootedUpper);
                if (!Finite(p.x) || !Finite(p.y)) throw new ArgumentException("Nonfinite patch vertex");
                b.final[i] = new Vector3(p.x,p.y,b.source[i].z);
                Current.maxPatchDisplacement=Mathf.Max(Current.maxPatchDisplacement, Vector3.Distance(b.final[i], b.source[i]));
            }
            for (int k=0; k<b.triangles.Length; k+=3)
            {
                int a=b.triangles[k], c=b.triangles[k+1], d=b.triangles[k+2];
                float before=Area(b.source[a],b.source[c],b.source[d]);
                if (Mathf.Abs(before)<1e-9f) continue;
                float ratio=Area(b.final[a],b.final[c],b.final[d])/before;
                Current.minTriangleAreaRatio=Mathf.Min(Current.minTriangleAreaRatio,ratio);
                if (ratio<0) Current.flippedTriangles++;
            }
        }
        // Geometry failures are retained in the evidence, not filtered from this diagnostic batch.
        foreach (var b in bindings)
        {
            b.renderer.Mesh.vertices=b.final;
            b.renderer.Mesh.RecalculateBounds();
            if (SandboxArmGeometry.IsHand(b.id) || b.id=="ArtMesh261")
            {
                float front=SandboxArmGeometry.IsHand(b.id)?pose.handFront:pose.forearmFront;
                if (!Finite(front) || front<0 || front>1) throw new ArgumentException("Invalid depth weight");
                b.renderer.transform.localPosition=b.position+Vector3.back*(.004f*(2*front-1));
                if(input.layerPolicy=="diagnostic-order")
                    b.renderer.MeshRenderer.sortingOrder=b.originalOrder+(front>=.5f?1000:-1000);
            }
        }
        if(input.layerPolicy=="upper-between-cap-sleeve")
        {
            // Narrow candidate for the identified upper-arm occluder. Keep the
            // shoulder ornament in front; forearm/hand retain their existing route.
            var upper=bindings.Single(b=>b.id=="ArtMesh263");
            var sleeve=bindings.Single(b=>b.id=="ArtMesh252");
            var cap=bindings.Single(b=>b.id=="ArtMesh250");
            upper.renderer.MeshRenderer.sortingOrder=sleeve.renderer.MeshRenderer.sortingOrder;
            var position=upper.renderer.transform.localPosition;
            position.z=(sleeve.renderer.transform.localPosition.z+cap.renderer.transform.localPosition.z)*.5f;
            upper.renderer.transform.localPosition=position;
        }
        Current.actualSortingOrders=bindings.Select(b=>b.renderer.MeshRenderer.sortingOrder).ToArray();
        if(input.layerPolicy=="comparison-visible")
        {
            foreach(var pair in distalOrders) pair.Key.MeshRenderer.sortingOrder=pair.Value+1000;
            foreach(var b in bindings.Where(b=>SandboxArmGeometry.IsHand(b.id)||b.id=="ArtMesh261"))
                b.renderer.MeshRenderer.sortingOrder=b.originalOrder+1000;
        }
        if(input.nativeDistalFront)
        {
            // Explicit foreground presentation for native distal arm only.
            // Not a depth-aware route; do not move the whole native clothing group.
            int front=model.Drawables.Select(d=>d.GetComponent<CubismRenderer>().MeshRenderer.sortingOrder).Max()+100;
            int lowest=distalOrders.Values.Min();
            foreach(var pair in distalOrders) pair.Key.MeshRenderer.sortingOrder=front+pair.Value-lowest;
        }
        Current.nativeDistalIds=distalOrders.Keys.Select(r=>r.GetComponent<CubismDrawable>().Id).ToArray();
        Current.nativeDistalSortingOrders=distalOrders.Keys.Select(r=>r.MeshRenderer.sortingOrder).ToArray();
        Current.attachmentControls=connection.AttachmentControls();
        Current.patchChain = connection.MaterialChain(id=>bindings.Single(b=>b.id==id).renderer.Mesh.vertices)
            .SelectMany(p=>new[]{p.x,p.y}).ToArray();
        Current.nativeChain = input.nativeMarkers.SelectMany(marker=>{
            var d=model.Drawables.Single(v=>v.Id==marker.mesh);
            var vertices=d.GetComponent<CubismRenderer>().Mesh.vertices;
            Vector3 p=Vector3.zero;
            if (marker.indices.Length!=marker.weights.Length) throw new ArgumentException("Marker size mismatch");
            for(int i=0;i<marker.indices.Length;i++) p+=vertices[marker.indices[i]]*marker.weights[i];
            return new[]{p.x,p.y};
        }).ToArray();
        if(Environment.GetEnvironmentVariable("ACTION_LAB_GEOMETRY_FEEDBACK")=="on")
        {
            Current.areaGradient=new float[3];
            Current.meshAreaRatios=MeasureAreas(pose);
            Current.meshAreaGradients=new float[bindings.Count*3];
            for(int j=0;j<3;j++)
            {
                var probe=new Pose { shoulder=pose.shoulder,elbow=pose.elbow,wrist=pose.wrist,
                    projection=pose.projection,girdle=pose.girdle };
                if(j==0) probe.shoulder+=.2f;
                if(j==1) probe.elbow+=.2f;
                if(j==2) probe.wrist+=.2f;
                var measured=MeasureAreas(probe);
                Current.areaGradient[j]=(measured.Min()-Current.minTriangleAreaRatio)/.2f;
                for(int k=0;k<measured.Length;k++)
                    Current.meshAreaGradients[k*3+j]=(measured[k]-Current.meshAreaRatios[k])/.2f;
            }
        }
    }

    public void Restore()
    {
        foreach(var pair in distalOrders) pair.Key.MeshRenderer.sortingOrder=pair.Value;
        float worst=0;
        foreach(var b in bindings)
        {
            var core=b.drawable.VertexPositions;
            // Restore both buffers, not only the last displayed mesh.
            for(int k=0;k<2;k++)
            {
                b.renderer.Mesh.vertices=core;
                b.renderer.Mesh.RecalculateBounds();
                b.renderer.SwapMeshes();
            }
            b.renderer.transform.localPosition=b.position;
            b.renderer.MeshRenderer.sortingOrder=b.originalOrder;
            var restored=b.renderer.Mesh.vertices;
            for(int i=0;i<core.Length;i++) worst=Mathf.Max(worst,Vector3.Distance(restored[i],core[i]));
        }
        if(Current!=null) Current.restoredMeshError=worst;
    }
}
