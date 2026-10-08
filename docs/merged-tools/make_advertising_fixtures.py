from pathlib import Path
import shutil
from PIL import Image,ImageDraw,ImageFont,ImageOps

out=Path('outputs/content-filter/examples')
def font(size,bold=False):
    return ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/TTF/DejaVuSans.ttf',size)

shutil.copy('outputs/waviness-tool/examples/straight-street.jpg',out/'street.jpg')
shutil.copy('outputs/waviness-tool/examples/1.png',out/'street-wavy.png')
shutil.copy('outputs/waviness-tool/examples/cows.png',out/'animals.png')
shutil.copy('outputs/waviness-tool/preview.jpg',out/'screenshot.jpg')

ad=Image.new('RGB',(1000,800),'#f6dc39');d=ImageDraw.Draw(ad)
d.text((70,55),'MEGA SALE',font=font(92,True),fill='#17231f')
d.text((75,195),'50% OFF',font=font(140,True),fill='#c13239')
d.rounded_rectangle((60,395,935,635),25,fill='#17231f')
d.text((120,427),'BUY NOW',font=font(98,True),fill='white')
d.text((155,555),'Free shipping · Limited offer',font=font(32),fill='white')
d.text((115,690),'SHOP EXAMPLE.COM TODAY',font=font(42,True),fill='#17231f')
ad.save(out/'advertisement.png')

street=Image.open(out/'street.jpg').convert('RGB')
ad=Image.new('RGB',(1000,1150),'#fff9ed');d=ImageDraw.Draw(ad)
d.text((65,40),'CITY BREAK SALE',font=font(76,True),fill='#183a31')
d.text((68,130),'DISCOVER YOUR NEXT DESTINATION',font=font(33),fill='#183a31')
ad.paste(ImageOps.fit(street,(920,700)),(40,210))
d.rounded_rectangle((42,930,956,1110),20,fill='#183a31')
d.text((78,957),'BOOK NOW · 50% OFF',font=font(55,True),fill='white')
d.text((79,1030),'Limited offer · Visit example.com',font=font(35),fill='white')
ad.save(out/'advertisement-with-street.png')

doc=Image.new('RGB',(1000,1300),'#fffef9');d=ImageDraw.Draw(doc)
d.text((80,60),'MEETING NOTES',font=font(48,True),fill='#202020')
for n in range(18):
 d.text((80,170+n*56),'This document describes project plans and tasks.',font=font(28),fill='#202020')
doc.save(out/'document.png')

logo=Image.new('RGB',(800,800),'#ffffff');d=ImageDraw.Draw(logo)
d.ellipse((150,130,650,630),fill='#335945');d.ellipse((250,230,550,530),fill='white')
d.text((190,665),'EXAMPLE',font=font(73,True),fill='#335945');logo.save(out/'logo.png')
Image.new('RGB',(640,480),'#cccccc').save(out/'blank.png')
print('Created content-filter examples.')
