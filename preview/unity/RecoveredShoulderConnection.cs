using System;
using System.Linq;
using UnityEngine;

/// <summary>Material-bound body/collar/socket attachment for the isolated mesh patch.</summary>
namespace ActionLab.Recovered {
public sealed class ShoulderConnection
{
    sealed class Point
    {
        public int[] ids;
        public Vector3 weights;
        public Vector2 Read(Vector3[] vertices) => (Vector2)(vertices[ids[0]]*weights.x+vertices[ids[1]]*weights.y+vertices[ids[2]]*weights.z);
    }
    readonly int bodyA,bodyB;
    readonly Point socketBinding,elbowBinding,wristBinding,palmBinding;
    readonly Vector2[] controls=new Vector2[4],targets=new Vector2[4];
    Vector2 socketBase,elbowBase,wristBase,palmBase,socket;
    SecondaryArmGirdle girdle;
    static float Cross(Vector2 a,Vector2 b) => a.x*b.y-a.y*b.x;
    static Point Bind(Vector3[] vertices,int[] triangles,Vector2 point)
    {
        Point best=null;float quality=float.NegativeInfinity;
        for(int i=0;i<triangles.Length;i+=3)
        {
            int a=triangles[i],b=triangles[i+1],c=triangles[i+2];
            Vector2 x=vertices[a],u=(Vector2)vertices[b]-x,v=(Vector2)vertices[c]-x,p=point-x;
            float area=Cross(u,v);if(Mathf.Abs(area)<1e-10f)continue;
            float wb=Cross(p,v)/area,wc=Cross(u,p)/area,wa=1-wb-wc;
            float score=Mathf.Min(wa,Mathf.Min(wb,wc));
            if(score>quality){quality=score;best=new Point{ids=new[]{a,b,c},weights=new Vector3(wa,wb,wc)};}
        }
        if(best==null || Mathf.Max(Mathf.Abs(best.weights.x),Mathf.Max(Mathf.Abs(best.weights.y),Mathf.Abs(best.weights.z)))>4)
            throw new ArgumentException("shoulder material binding outside safe neighbourhood");
        return best;
    }
    public ShoulderConnection(Vector3[] skin,int[] skinTriangles,Vector3[] fore,int[] foreTriangles,Vector3[] hand,int[] handTriangles,
        Vector2[] correctedNodes=null)
    {
        bodyA=Enumerable.Range(0,skin.Length).OrderByDescending(i=>skin[i].x).First();
        bodyB=Enumerable.Range(0,skin.Length).OrderByDescending(i=>skin[i].x).Skip(1).First();
        if(correctedNodes==null)
        {
            int socketIndex=Enumerable.Range(0,skin.Length).OrderBy(i=>((Vector2)skin[i]-SandboxArmGeometry.Shoulder).sqrMagnitude).First();
            socketBinding=new Point { ids=new[]{socketIndex,socketIndex,socketIndex},weights=new Vector3(1,0,0) };
        }
        else socketBinding=Bind(skin,skinTriangles,correctedNodes[0]);
        elbowBinding=Bind(fore,foreTriangles,correctedNodes==null?SandboxArmGeometry.Elbow:correctedNodes[1]);
        wristBinding=Bind(hand,handTriangles,correctedNodes==null?SandboxArmGeometry.Wrist:correctedNodes[2]);
        palmBinding=Bind(hand,handTriangles,correctedNodes==null?new Vector2(-.1972895f,.007617678f):correctedNodes[3]);
    }
    public void Update(Vector3[] skin,Vector3[] fore,Vector3[] hand,SecondaryArmGirdle value)
    {
        girdle=value;socketBase=socketBinding.Read(skin);elbowBase=elbowBinding.Read(fore);
        wristBase=wristBinding.Read(hand);palmBase=palmBinding.Read(hand);
        controls[0]=skin[bodyA];controls[1]=skin[bodyB];controls[2]=socketBase;controls[3]=elbowBase;
        socket=socketBase+girdle.SocketOffset;
        targets[0]=controls[0]+girdle.RootOffset;targets[1]=controls[1]+girdle.RootOffset;targets[2]=socket;targets[3]=elbowBase;
    }
    public static Vector2 RigidMls(Vector2 v,Vector2[] source,Vector2[] destination)
    {
        var weights=new float[source.Length];float total=0;Vector2 p=Vector2.zero,q=Vector2.zero;
        for(int i=0;i<source.Length;i++)
        {
            float d=(source[i]-v).sqrMagnitude;if(d<1e-12f)return destination[i];
            weights[i]=1/d;total+=weights[i];
        }
        for(int i=0;i<source.Length;i++){weights[i]/=total;p+=source[i]*weights[i];q+=destination[i]*weights[i];}
        float real=0,imag=0;
        for(int i=0;i<source.Length;i++)
        {Vector2 a=source[i]-p,b=destination[i]-q;real+=weights[i]*Vector2.Dot(a,b);imag+=weights[i]*Cross(a,b);}
        float norm=Mathf.Sqrt(real*real+imag*imag);
        if(norm<1e-12f)throw new ArgumentException("indeterminate shoulder attachment rotation");
        Vector2 d0=v-p;
        return q+new Vector2(real*d0.x-imag*d0.y,imag*d0.x+real*d0.y)/norm;
    }
    public Vector2 Deform(Vector2 v,Vector2 rest,string id,float shoulder,float elbow,float wrist,SecondaryArmProjection projection,bool rootedUpper=false)
    {
        if(shoulder==0 && elbow==0 && wrist==0 && projection.IsIdentity && girdle.IsIdentity)return v;
        Vector2 ep=socket+SandboxArmGeometry.Rotate((elbowBase-socketBase)*projection.upperLength,shoulder);
        Vector2 wp=ep+SandboxArmGeometry.Rotate((wristBase-elbowBase)*projection.forearmLength,shoulder+elbow);
        Vector2 palm=(palmBase-wristBase).normalized,side=new Vector2(-palm.y,palm.x),delta=v-wristBase;
        float along=Vector2.Dot(delta,palm),across=Vector2.Dot(delta,side);
        Vector2 projected=palm*(along*projection.palmLength+across*projection.palmShear)+side*(across*projection.palmWidth);
        if(SandboxArmGeometry.IsHand(id))return wp+SandboxArmGeometry.Rotate(projected,shoulder+elbow+wrist);
        if(id=="ArtMesh261")
        {
            float weight=SandboxArmGeometry.CuffWeight(rest);Vector2 axis=(wristBase-elbowBase).normalized;
            Vector2 scaled=delta+axis*(Vector2.Dot(delta,axis)*(projection.forearmLength-1));
            return wp+SandboxArmGeometry.Rotate(Vector2.Lerp(scaled,projected,weight),shoulder+elbow+weight*wrist);
        }
        targets[0]=controls[0]+girdle.RootOffset;targets[1]=controls[1]+girdle.RootOffset;
        targets[2]=socket;targets[3]=ep;
        if(rootedUpper)
        {
            // Candidate: retain the material-root attachment, but rotate the upper
            // segment about the same socket used to construct the forearm elbow.
            // Distal vertices use the complete upper transform, not a rigid MLS fit.
            float weight=SandboxArmGeometry.RootWeight(rest);
            var rootTargets=new[]{targets[0],targets[1],socket,elbowBase};
            Vector2 attached=RigidMls(v,controls,rootTargets);
            Vector2 axis=(elbowBase-socketBase).normalized,local=attached-socket;
            local+=axis*(Vector2.Dot(local,axis)*(projection.upperLength-1)*weight);
            return socket+SandboxArmGeometry.Rotate(local,shoulder*weight);
        }
        return RigidMls(v,controls,targets);
    }
    public object Observe(Func<string,Vector3[]> source,Func<string,Vector3[]> rest,Func<string,Vector3[]> final,SecondaryArmProjection projection)
    {
        float[] Row(Point point,string mesh,bool neutral)
        {
            Vector2 value;
            if(!neutral)value=point.Read(final(mesh));
            else
            {
                var src=source(mesh);var reference=rest(mesh);value=Vector2.zero;
                for(int i=0;i<3;i++)value+=Deform(src[point.ids[i]],reference[point.ids[i]],mesh,0,0,0,projection)*point.weights[i];
            }
            return new[]{value.x,value.y};
        }
        object Chain(bool neutral)=>new{shoulder=Row(socketBinding,"ArtMesh251",neutral),elbow=Row(elbowBinding,"ArtMesh261",neutral),wrist=Row(wristBinding,"ArtMesh253",neutral),palm=Row(palmBinding,"ArtMesh253",neutral)};
        return new{actual=Chain(false),neutral=Chain(true),interpretation="same-frame barycentric material chains, not anatomical joints"};
    }
    public Vector2[] MaterialChain(Func<string,Vector3[]> final) => new[] {
        socketBinding.Read(final("ArtMesh251")), elbowBinding.Read(final("ArtMesh261")),
        wristBinding.Read(final("ArtMesh253")), palmBinding.Read(final("ArtMesh253")) };
    public object Readback() => new { interpretation="material attachment controls; not anatomical joint truth",
        bodyVertexIndices=new[]{bodyA,bodyB},socketVertexIndex=socketBinding.ids[0],bodyA=new[]{targets[0].x,targets[0].y},bodyB=new[]{targets[1].x,targets[1].y},
        socket=new[]{socket.x,socket.y},socketBase=new[]{socketBase.x,socketBase.y},girdle };
    public Vector2[] AttachmentControls() => controls.Concat(targets).ToArray();
}

}
