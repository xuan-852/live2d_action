using UnityEngine;
namespace ActionLab.Recovered {
/// <summary>Positive, bounded frontal projection; no palm/back reflection or new texture.</summary>
[System.Serializable] public struct SecondaryArmProjection
{
    public float upperLength;
    public float forearmLength;
    public float palmLength;
    public float palmWidth;
    public float palmShear;
    public static SecondaryArmProjection Identity => new SecondaryArmProjection { upperLength=1,forearmLength=1,palmLength=1,palmWidth=1 };
    static bool In(float x,float a,float b) => !float.IsNaN(x)&&!float.IsInfinity(x)&&x>=a&&x<=b;
    public bool Valid => In(upperLength,.75f,1)&&In(forearmLength,.55f,1)&&In(palmLength,.55f,1)&&In(palmWidth,.35f,1)&&In(palmShear,-.2f,.2f);
    // Version 4: shared source-arm scale may lengthen the originally short upper
    // segment. Actual triangles are still checked before any mesh is committed.
    public bool ReachValid => In(upperLength,.75f,1.7f)&&In(forearmLength,.55f,1)&&In(palmLength,.55f,1)&&In(palmWidth,.35f,1)&&In(palmShear,-.2f,.2f);
    public bool IsIdentity => upperLength==1&&forearmLength==1&&palmLength==1&&palmWidth==1&&palmShear==0;
    public static SecondaryArmProjection Lerp(SecondaryArmProjection a,SecondaryArmProjection b,float w) => new SecondaryArmProjection {
        upperLength=Mathf.Lerp(a.upperLength,b.upperLength,w),forearmLength=Mathf.Lerp(a.forearmLength,b.forearmLength,w),
        palmLength=Mathf.Lerp(a.palmLength,b.palmLength,w),palmWidth=Mathf.Lerp(a.palmWidth,b.palmWidth,w),palmShear=Mathf.Lerp(a.palmShear,b.palmShear,w) };
}
[System.Serializable] public struct SecondaryArmGirdle
{
    public float rootX;
    public float rootY;
    public float socketX;
    public float socketY;
    static bool Bounded(float v) => !float.IsNaN(v)&&!float.IsInfinity(v)&&Mathf.Abs(v)<=.05f;
    public bool Valid => Bounded(rootX)&&Bounded(rootY)&&Bounded(socketX)&&Bounded(socketY);
    public bool IsIdentity => rootX==0&&rootY==0&&socketX==0&&socketY==0;
    public Vector2 RootOffset => new Vector2(rootX,rootY);
    public Vector2 SocketOffset => new Vector2(socketX,socketY);
    public static SecondaryArmGirdle Lerp(SecondaryArmGirdle a,SecondaryArmGirdle b,float w) => new SecondaryArmGirdle {
        rootX=Mathf.Lerp(a.rootX,b.rootX,w),rootY=Mathf.Lerp(a.rootY,b.rootY,w),
        socketX=Mathf.Lerp(a.socketX,b.socketX,w),socketY=Mathf.Lerp(a.socketY,b.socketY,w) };
}
}
