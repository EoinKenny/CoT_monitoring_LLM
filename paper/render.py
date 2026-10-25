"""Render figures from numeric inputs, then compose the seven-page reference layout."""
import os,runpy
from pathlib import Path
from pypdf import PdfReader,PdfWriter,Transformation

def place(page,path,x,y,width,height):
 source=PdfReader(path).pages[0]
 sw,sh=float(source.mediabox.width),float(source.mediabox.height)
 scale=min(width/sw,height/sh)
 page.merge_transformed_page(source,Transformation().scale(scale).translate(x+(width-sw*scale)/2,y+(height-sh*scale)/2))

def render(data,out,appendix=True):
 import matplotlib.pyplot as plt
 out.mkdir(parents=True,exist_ok=True)
 os.environ['NN_PLOT_DATA']=str(data);os.environ['NN_PLOT_OUTPUT']=str(out)
 for name in ['performance','ablation','labels','contrast','traces','manifold']+(['appendix'] if appendix else []):
  plt.rcdefaults()
  print('Plotting',name,flush=True)
  runpy.run_module('paper.plots.'+name,run_name='__main__')
 compose(out)

def compose(out):
 import matplotlib.pyplot as plt
 # Numeric panels remain vector graphics in the combined PDF.
 writer=PdfWriter()
 page=writer.add_blank_page(width=959.76,height=540)
 place(page,out/'overall_four_panel.pdf',45,222,825,225)
 page=writer.add_blank_page(width=959.76,height=540)
 place(page,out/'overall_metrics.pdf',120,160,660,285)
 page=writer.add_blank_page(width=959.76,height=540)
 place(page,out/'label_prevalence_judge_dataset.pdf',25,160,500,280)
 place(page,out/'overall_contrast_net_square.pdf',530,148,295,295)
 page=writer.add_blank_page(width=959.76,height=540)
 place(page,out/'manifold_background.pdf',145,185,585,250)
 fig=plt.figure(figsize=(13.33,7.5));fig.text(.5,.5,'APPENDIX',ha='center',va='center',fontsize=28)
 fig.savefig(out/'appendix_divider.pdf');plt.close(fig)
 page=writer.add_blank_page(width=959.76,height=540);place(page,out/'appendix_divider.pdf',0,0,959.76,540)
 page=writer.add_blank_page(width=959.76,height=540);place(page,out/'genre_f1_differences.pdf',100,155,770,290)
 page=writer.add_blank_page(width=959.76,height=540);place(page,out/'agreement_matrices.pdf',115,145,720,340)
 with (out/'figures.pdf').open('wb') as f:writer.write(f)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Reproduced paper figures</title><style>body{font:17px system-ui;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}</style><h1>Paper figures</h1><p><a href="figures.pdf">Combined PDF</a></p>'+''.join(f'<h2>{label}</h2><img src="{name}.png"><p><a href="{name}.pdf">PDF</a></p>' for name,label in [('overall_four_panel','Performance'),('overall_metrics','Ablations'),('label_prevalence_judge_dataset','Mechanism labels'),('overall_contrast_net_square','Exploratory contrast'),('manifold_background','Schematic manifold'),('genre_f1_differences','Ablations by genre'),('agreement_matrices','Written explanation agreement')]))
