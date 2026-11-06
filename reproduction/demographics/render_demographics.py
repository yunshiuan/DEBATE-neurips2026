"""Render supplied-708-session demographic aggregates without changing the paper repo."""
import os, tempfile, argparse
os.environ.setdefault('MPLCONFIGDIR',tempfile.gettempdir()+'/debate-matplotlib')
import hashlib,json,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--aggregates',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
ROOT=args.output_dir
ROOT.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
PREVIEW=ROOT/'previews'; PREVIEW.mkdir(exist_ok=True)
ALT=ROOT/'alternatives'; ALT.mkdir(exist_ok=True)
AGG=json.loads(args.aggregates.read_text())
ORIGINAL={'demo_age_fig.pdf': {'points': [432.0, 190.0]}, 'demo_bible_fig.pdf': {'points': [432.0, 243.0]}, 'demo_children_fig.pdf': {'points': [432.0, 243.0]}, 'demo_education_fig.pdf': {'points': [432.0, 243.0]}, 'demo_ethnicity_fig.pdf': {'points': [432.0, 243.0]}, 'demo_evangelical_fig.pdf': {'points': [432.0, 243.0]}, 'demo_gender_fig.pdf': {'points': [432.0, 243.0]}, 'demo_income_fig.pdf': {'points': [432.0, 243.0]}, 'demo_marriage_fig.pdf': {'points': [432.0, 243.0]}, 'demo_occupuation_fig.pdf': {'points': [432.0, 155.0]}, 'demo_origin_fig.pdf': {'points': [432.0, 135.0]}, 'demo_political_party_fig.pdf': {'points': [432.0, 243.0]}, 'demo_political_view_fig.pdf': {'points': [432.0, 243.0]}, 'demo_religion_fig.pdf': {'points': [432.0, 243.0]}, 'demo_residence_fig.pdf': {'points': [432.0, 243.0]}, 'demo_residency_fig.pdf': {'points': [432.0, 333.0]}}
ORDER={
 'gender':['male','female','non-binary','prefer not to answer'],
 'ethnicity':['black','hispanic','white','asian','native-american','other','prefer not to answer'],
 'education':['high-school','associate','professional','some-college','bachelor','master-doctoral','prefer not to answer'],
 'income':['less-25k','25k-50k','50k-75k','75k-100k','100k-150k','150k-200k','over-200k','prefer not to answer'],
 'occupation':['finance','other','engineering','health-care','sales','education','manufacturing','media','prefer not to answer','construction','arts','restaurant','transportation','research','law','government','agriculture','insurance','wholesale','personal-care','real-estate','non-profit','cleaning','maintenance','mining'],
 'politicalIdentity':['strong-republican','republican','independent','democrat','strong-democrat','other','prefer not to answer'],
 'politicalViews':['very-conservative','conservative','moderate','liberal','very-liberal','prefer not to answer','dont-know'],
 'bibleBelief':['literal','inspired','ancient-book','prefer not to answer'],
 'evangelical':['yes','no','prefer not to answer','dont-know'],
 'religion':['roman-catholic','protestant','no-religion','other-christian','prefer not to answer','other-non-christian','muslim','jewish','dont-know','mormon'],
 'maritalStatus':['married','never-married','living-partner','divorced','separated','prefer not to answer','widowed'],
 'childrenSchool':['public','private','home','university','out-of-school','no-children','prefer not to answer'],
 'residence':['urban','suburban','rural','prefer not to answer','dont-know'],
}
TITLE={'age':'Age','gender':'Gender','ethnicity':'Ethnicity','education':'Education','income':'Income','occupation':'Occupation','politicalIdentity':'Political identity','politicalViews':'Political views','bibleBelief':'Bible belief','evangelical':'Evangelical','religion':'Religion','maritalStatus':'Marital status','childrenSchool':"Children’s schooling",'residence':'Residence','origin':'Country of origin'}
ORIGIN_SHORT={'United States of America':'United States','Prefer not to answer':'Prefer not to answer','United States Minor Outlying Islands':'U.S. outlying islands','Korea, Republic of':'South Korea','United Kingdom of Great Britain and Northern Ireland':'United Kingdom','Virgin Islands, U.S.':'U.S. Virgin Islands','Venezuela, Bolivarian Republic of':'Venezuela','Antigua and Barbuda':'Antigua & Barbuda','United Arab Emirates':'U.A.E.'}
BLUE='#92AFE0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.edgecolor':'#4B4B4B','axes.linewidth':0.65,'xtick.major.width':0.6,'ytick.major.width':0.6,'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
manifest=[]
layout_issues=[]

def finish(fig,stem,record,alternative=False):
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer()
 w,h=fig.bbox.width,fig.bbox.height
 outside=[]
 for ax in fig.axes:
  xlabels=[obj for pos,obj in zip(ax.get_xticks(),ax.get_xticklabels()) if ax.get_xlim()[0]<=pos<=ax.get_xlim()[1]]
  ylabels=[obj for pos,obj in zip(ax.get_yticks(),ax.get_yticklabels()) if ax.get_ylim()[0]<=pos<=ax.get_ylim()[1]]
  for obj in [ax.title,ax.xaxis.label,ax.yaxis.label,*xlabels,*ylabels,*ax.texts]:
   if not obj.get_text(): continue
   b=obj.get_window_extent(renderer)
   if b.x0 < -0.6 or b.y0 < -0.6 or b.x1>w+0.6 or b.y1>h+0.6:outside.append(obj.get_text())
 if outside:layout_issues.append({'figure':stem,'alternative':alternative,'outside_page':outside})
 dest=ALT if alternative else OUT
 path=dest/(stem+'.pdf')
 fig.savefig(path,metadata={'Title':record['title'],'Author':'DEBATE','Subject':'Demographic distributions from the supplied 708-session dataset'})
 png=(ALT if alternative else PREVIEW)/(stem+'.png')
 fig.savefig(png,dpi=220)
 record.update(pdf=str(path),png=str(png),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),page_size_points=ORIGINAL[stem+'.pdf']['points'],outside_page_text=outside)
 manifest.append(record)
 plt.close(fig)

for stem,field in AGG['figure_field_mapping'].items():
 data=AGG['fields'][field]; cats=data['categories']
 w,h=ORIGINAL[stem+'.pdf']['points']
 fig=plt.figure(figsize=(w/72,h/72))
 record={'stem':stem,'field':field,'title':TITLE[field],'response_n':data['response_n'],'multi_select':data['multi_select'],'aggregation':'none','categories':cats}
 if field=='age':
  ax=fig.add_axes([.092,.16,.89,.71])
  values=np.array([x['value'] for x in cats]);weights=np.array([x['count'] for x in cats])
  # Original figure used 15 equal-width bins over the observed age range.
  counts,edges=np.histogram(values,bins=15,range=(data['minimum'],data['maximum']),weights=weights)
  widths=np.diff(edges)
  bars=ax.bar(edges[:-1],counts,width=widths,align='edge',color=BLUE,edgecolor='#2F2F2F',linewidth=.65)
  for p,count in zip(bars,counts):
   ax.annotate(f'{count}\n({count/data["response_n"]*100:.1f}%)',(p.get_x()+p.get_width()/2,p.get_height()),xytext=(0,2),textcoords='offset points',ha='center',va='bottom',fontsize=6.3,linespacing=1.05)
  ax.set_ylim(0,max(counts)*1.19)
  ax.set_xlabel('Age (years)');ax.set_ylabel('Count')
  ax.set_title(f'Age (mean={data["mean"]:.2f}, SD={data["sample_sd"]:.2f}, min={data["minimum"]:.0f}, max={data["maximum"]:.0f})',fontsize=9.3,pad=5)
  record['histogram']={'bins':15,'edges':edges.tolist(),'counts':counts.tolist(),'percent':[float(n/data['response_n']*100) for n in counts]}
 else:
  by_label={c['label']:c for c in cats}
  order=ORDER.get(field,[c['label'] for c in cats])
  assert set(order)==set(by_label),(field,set(order)^set(by_label))
  cats=[by_label[label] for label in order]
  count=np.array([c['count'] for c in cats]); n=len(cats)
  if field=='occupation':
   ax=fig.add_axes([.085,.40,.90,.47]); fs=5.8; rot=52; annotation_fs=5.1
   labels=[c['label'].replace('prefer not to answer','Prefer not to answer') for c in cats]
  elif field=='origin':
   ax=fig.add_axes([.085,.45,.90,.42]); fs=4.8; rot=90; annotation_fs=4.3
   labels=[ORIGIN_SHORT.get(c['label'],c['label']) for c in cats]
  else:
   ax=fig.add_axes([.092,.34,.89,.55]); fs=8; rot=45; annotation_fs=7.2 if n<8 else 6.8
   labels=[c['label'] for c in cats]
  bars=ax.bar(range(n),count,color=BLUE,edgecolor='white',linewidth=.5,width=.8)
  for p,c in zip(bars,cats):
   if field=='origin':
    # Preserve all 47 categories; omit tiny numeric annotations that overlap.
    if c['count']==max(count): label=f'{c["count"]:,} ({c["percent"]:.1f}%)'
    else: continue
   elif field=='occupation': label=f'{c["count"]}\n({c["percent"]:.1f}%)'
   elif n>=8: label=f'{c["count"]:,}\n({c["percent"]:.1f}%)'
   else: label=f'{c["count"]:,} ({c["percent"]:.1f}%)'
   ax.annotate(label,(p.get_x()+p.get_width()/2,p.get_height()),xytext=(0,2),textcoords='offset points',ha='center',va='bottom',fontsize=annotation_fs,linespacing=1.05)
  assert len(labels)==len(set(labels)), (field,'ambiguous display labels')
  ax.set_xticks(range(n),labels,rotation=rot,ha='right' if rot!=90 else 'center',rotation_mode='anchor' if rot!=90 else 'default',fontsize=fs)
  ax.tick_params(axis='x',length=0,pad=5)
  ax.tick_params(axis='y',labelsize=6 if field in ['origin','occupation'] else 8)
  ax.set_ylim(0,max(count)*(1.22 if field in ['occupation','origin'] or n>=8 else 1.13))
  ax.set_xlim(-.65,n-.35)
  ax.set_ylabel('Count',fontsize=7 if field in ['origin','occupation'] else 8)
  if field not in ['origin','occupation']:ax.set_xlabel(TITLE[field],labelpad=4)
  ax.set_title(TITLE[field]+' distribution',fontsize=8.5 if field in ['origin','occupation'] else 10,pad=5)
  record['categories']=[dict(c,display_label=label) for c,label in zip(cats,labels)]
  record['order_rule']='Original category order preserved' if field in ORDER else 'Descending response count (as in original style); original exact country order cannot be recovered from low-resolution raster.'
 finish(fig,stem,record)

# Contact sheet is only a review aid; manuscript assets are vector PDFs.
canvas=Image.new('RGB',(2100,1820),'white');draw=ImageDraw.Draw(canvas)
try:
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',21)
except OSError:
 font=ImageFont.load_default()
for i,record in enumerate(manifest[:15]):
 im=Image.open(record['png']).convert('RGB')
 thumb=ImageOps.contain(im,(680,320))
 x=(i%3)*700;y=(i//3)*364
 draw.text((x+14,y+7),record["title"],font=font,fill='#222222')
 canvas.paste(thumb,(x+(700-thumb.width)//2,y+35))
canvas.save(ROOT/'contact_sheet_20261105.png')
output={'source_aggregates':str(args.aggregates),'source_aggregate_sha256':hashlib.sha256(args.aggregates.read_bytes()).hexdigest(),'render_script':str(Path(__file__)),'pdfs':manifest,'layout_issues':layout_issues,'default_asset_count':15,'alternative_asset_count':0,'notes':['Outputs are rendered from aggregate demographic counts.','All15 original PDF page sizes are preserved exactly.','PDFs contain vector bars and text; original PDFs contained raster images.','Ethnicity and children schooling are multiple-response items; percentages use per-item respondent denominators.','Age uses15 equal-width bins and the18–120valid-age rule from the audited aggregates.','All category counts are preserved by default; only display labels are shortened for some country names.']}
(ROOT/'render_manifest_20261105.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'figures':15,'alternative':0,'layout_issues':layout_issues,'contact_sheet':str(ROOT/'contact_sheet_20261105.png')},indent=2))
