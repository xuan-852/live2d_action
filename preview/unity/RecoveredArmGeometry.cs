using UnityEngine;
namespace ActionLab.Recovered {
public static class SandboxArmGeometry {
    public static readonly string[] TargetIds = { "ArtMesh250", "ArtMesh251", "ArtMesh252", "ArtMesh253",
        "ArtMesh254", "ArtMesh255", "ArtMesh256", "ArtMesh257", "ArtMesh258", "ArtMesh259", "ArtMesh261", "ArtMesh262", "ArtMesh263",
        "ArtMesh269", "ArtMesh270", "ArtMesh271" };
    /// <summary>Cloth that must keep its original Cubism-driven deformation.</summary>
    public static readonly string[] ClothIds = { "ArtMesh303", "ArtMesh325", "ArtMesh326", "ArtMesh327",
        "ArtMesh328", "ArtMesh329", "ArtMesh330" };
    public static readonly Vector2 Shoulder = new Vector2(-.105f, .310f);
    public static readonly Vector2 Elbow = new Vector2(-.145f, .205f);
    public static readonly Vector2 Wrist = new Vector2(-.199f, .025f);
    /// <summary>
    /// Shoulder-root blend measured along the shoulder-&gt;elbow axis, in upper-arm
    /// lengths from the root. Below Start a vertex stays with the torso; from End
    /// (the elbow) outward it follows the arm completely. The ramp is linear: a
    /// smoothstep of the same width lost 9 deg of usable shoulder range.
    /// </summary>
    /// <remarks>
    /// Measured on the 16 real target meshes (2 011 triangles). The previous rest.y
    /// band lerped *positions* over .28..345, collapsed the sleeve to a 0.544 area
    /// ratio at -15 deg, and flipped from -35 deg — exactly the range the source
    /// clips needed. This band scales the *angle* instead, so rigid edges survive:
    /// 0.87 at -15 deg, 0.53 at -55 deg, no flip anywhere in the envelope. It also
    /// spans the whole shoulder cap rather than one thin strip, which is what the
    /// "shoulder is where it stops" report was about.
    /// </remarks>
    public const float RootBlendStart = -.6f;
    public const float RootBlendEnd = 1f;
    static readonly Vector2 UpperAxis = (Elbow - Shoulder).normalized;
    public static readonly float UpperLength = (Elbow - Shoulder).magnitude;
    /// <summary>
    /// Externally addressable pose envelope in degrees (x=shoulder, y=elbow, z=wrist).
    /// Each bound is where the reworked deform first dropped a target triangle below
    /// a 0.5 area ratio or flipped it, with margin: shoulder -57..+74 measured, held
    /// to ±56 for symmetry; wrist negative is bounded by anatomy rather than by
    /// geometry (the cuff still measures 0.71 at -110), so it stops at -100 where
    /// the source clips actually reach; wrist positive pinches the cuff at +36.
    /// Both source cards now fit inside with no clipping.
    /// </summary>
    public static readonly Vector3 PoseMin = new Vector3(-56f, 0f, -100f);
    public static readonly Vector3 PoseMax = new Vector3(56f, 145f, 30f);
    public static Vector2 Rotate(Vector2 v, float degrees)
    {
        float r=degrees*Mathf.Deg2Rad,c=Mathf.Cos(r),s=Mathf.Sin(r);
        return new Vector2(c*v.x-s*v.y,s*v.x+c*v.y);
    }
    /// <summary>Rigid hand chain: shoulder + elbow + wrist rotate it as one piece.</summary>
    public static bool IsHand(string id) => id=="ArtMesh254" || id=="ArtMesh255" || id=="ArtMesh256"
        || id=="ArtMesh257" || id=="ArtMesh258" || id=="ArtMesh259" || id=="ArtMesh253";
    /// <summary>
    /// Blend weight of the shoulder rotation for one rest vertex, 0 = torso, 1 = arm.
    /// Measured along the upper-arm axis rather than by screen height, so the sleeve
    /// cap keeps its own proportions while it turns.
    /// </summary>
    public static float RootWeight(Vector2 rest)
    {
        float t=Vector2.Dot(rest-Shoulder,UpperAxis)/UpperLength;
        return Mathf.Clamp01((t-RootBlendStart)/(RootBlendEnd-RootBlendStart));
    }
    /// <summary>Blend weight of the wrist bend across the forearm cuff, 0 = elbow side.</summary>
    /// <remarks>
    /// The old band (.065..025) applied the bend over a strip that is narrower than the
    /// sleeve overlap, so the cuff pinched: 0.511 area ratio at +20 deg. Spreading it
    /// over .09..010 keeps 0.725 there and still 0.773 at -85 deg.
    /// </remarks>
    public static float CuffWeight(Vector2 rest) =>
        Mathf.SmoothStep(0,1,Mathf.InverseLerp(.09f,.01f,rest.y));
    public static Vector2 Deform(Vector2 v, Vector2 rest, string id, Vector2 offset, float shoulder, float elbow, float wrist)
        => Deform(v,rest,id,offset,shoulder,elbow,wrist,SecondaryArmProjection.Identity);
    static Vector2 Along(Vector2 v,Vector2 axis,float scale) => v+axis*(Vector2.Dot(v,axis)*(scale-1));
    static readonly Vector2 ForeAxis=(Wrist-Elbow).normalized;
    static readonly Vector2 PalmAxis=(new Vector2(-.1972895f,.007617678f)-Wrist).normalized;
    static Vector2 PalmProject(Vector2 q,SecondaryArmProjection p)
    {
        Vector2 side=new Vector2(-PalmAxis.y,PalmAxis.x);
        float along=Vector2.Dot(q,PalmAxis),across=Vector2.Dot(q,side);
        return PalmAxis*(along*p.palmLength+across*p.palmShear)+side*(across*p.palmWidth);
    }
    public static float ProjectionAreaFloor(string id,SecondaryArmProjection p)
        => IsHand(id)?p.palmLength*p.palmWidth:id=="ArtMesh261"?Mathf.Min(p.forearmLength,p.palmLength*p.palmWidth):p.upperLength;
    public static Vector2 Deform(Vector2 v, Vector2 rest, string id, Vector2 offset, float shoulder, float elbow, float wrist,SecondaryArmProjection p)
    {
        if (shoulder==0 && elbow==0 && wrist==0 && p.IsIdentity) return v;
        Vector2 s=Shoulder+offset,e=Elbow+offset,w=Wrist+offset;
        Vector2 ep=s+Rotate((e-s)*p.upperLength,shoulder),wp=ep+Rotate((w-e)*p.forearmLength,shoulder+elbow);
        Vector2 hand=wp+Rotate(PalmProject(v-w,p),shoulder+elbow+wrist);
        if (IsHand(id)) return hand;
        if (id=="ArtMesh261")
        {
            // Forearm stays rigid through the elbow; sleeve overlap covers the joint.
            // The wrist bend is blended as an *angle* near the cuff, because lerping
            // positions there pinched the sleeve as soon as the hand bent backwards.
            float ww=CuffWeight(rest);
            Vector2 local=Vector2.Lerp(Along(v-w,ForeAxis,p.forearmLength),PalmProject(v-w,p),ww);
            return wp+Rotate(local,shoulder+elbow+ww*wrist);
        }
        // Shoulder root: scale the rotation angle across the cap instead of lerping
        // between the undeformed and the fully rotated position. Position lerp
        // collapsed every edge that straddled the old band, which is what produced
        // the 0.54 area ratio and the -35 deg fold.
        float weight=RootWeight(rest);
        return s+Rotate(Along(v-s,UpperAxis,Mathf.Lerp(1,p.upperLength,weight)),shoulder*weight);
    }
}
}
