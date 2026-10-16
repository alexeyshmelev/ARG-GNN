from pathlib import Path
import subprocess,io,base64,sys
from PIL import Image,ImageOps,ImageDraw
pdf='/workspace-SR003.nfs2/ARG-GNN/paper/build/current_full_panel_FLARE_20261016_v1/current_full_panel_working.pdf'
pages=[]
for number in range(int(sys.argv[1]),int(sys.argv[2])+1):
 r=subprocess.run(['pdftoppm','-r','60','-f',str(number),'-l',str(number),'-singlefile','-png',pdf],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
 page=Image.open(io.BytesIO(r.stdout)).convert('RGB')
 panel=Image.new('RGB',(page.width,page.height+22),'#dbe1e8')
 panel.paste(page,(0,22));ImageDraw.Draw(panel).text((8,4),f'Page {number}',fill='black')
 pages.append(panel)
out=Image.new('RGB',(sum(p.width for p in pages),max(p.height for p in pages)),'white')
x=0
for page in pages:out.paste(page,(x,0));x+=page.width
buffer=io.BytesIO();out.save(buffer,format='JPEG',quality=55,optimize=True)
print(base64.b64encode(buffer.getvalue()).decode())
