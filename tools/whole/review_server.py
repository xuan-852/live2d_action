"""Optional localhost review server with MP4 range support; not needed offline."""
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from functools import partial
import sys

class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        path=self.translate_path(self.path)
        if 'Range' not in self.headers or not self.path.endswith('.mp4'):
            self.remaining=None;return super().send_head()
        f=open(path,'rb');f.seek(0,2);size=f.tell()
        value=self.headers['Range'].replace('bytes=','').split(',')[0]
        left,right=value.split('-');start=int(left or 0);end=min(int(right) if right else size-1,size-1)
        f.seek(start);self.remaining=end-start+1
        self.send_response(206);self.send_header('Content-Type','video/mp4')
        self.send_header('Accept-Ranges','bytes');self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(self.remaining));self.end_headers();return f
    def copyfile(self,source,outputfile):
        if getattr(self,'remaining',None) is None:return super().copyfile(source,outputfile)
        while self.remaining>0:
            data=source.read(min(self.remaining,65536))
            if not data:break
            outputfile.write(data);self.remaining-=len(data)

if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8768),partial(Handler,directory=sys.argv[1])).serve_forever()
