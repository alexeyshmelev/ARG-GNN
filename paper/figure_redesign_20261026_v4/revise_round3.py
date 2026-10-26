from pathlib import Path
p=Path(__file__).with_name('build_figures.py')
s=p.read_text().replace('canvas(12.2,6.10)', 'canvas(12.7,6.35)')
a=s.index('# Edges have a distinct path')
b=s.index("text(.16,2.22,'b'",a)
s=s[:a]+'''# Separate tree and variant inputs explicitly form learned edge gates.
arrow([(4.34,.80),(4.43,.80),(4.43,3.82),(4.91,3.82)])
arrow([(5.13,1.00),(4.59,1.00),(4.59,4.00),(4.91,4.00)])
# White underpasses mark crossings with the returning leaf-feature line.
for xx in [4.43,4.59]:ax.plot([xx,xx],[3.20,3.38],color='white',lw=2.6,zorder=3)
line([(4.72,3.29),(4.30,3.29)],.75)
block(4.91,3.59,3.50,.61,'Edge gates\\nBranch geometry + allele agreement',True,5.9)
arrow([(6.66,4.20),(6.66,4.42),(4.05,4.42),(4.05,4.54)])
DY=1.6
''' + s[b:]
# Extend the base input to its shifted all-node matrix.
s=s.replace('arrow([(2.35,3.57),(2.35,3.98)])','arrow([(2.35,3.57),(2.35,4.48)])')
a=s.index("text(11.72,3.91,'Compare preceding")
b=s.index('arrow([(11.72,3.24)',a)
s=s[:a]+'''text(11.72,3.91,'Presence in adjacent trees → flags',5.8)
for j,label in enumerate(['$T_{t-1}$','$T_t$','$T_{t+1}$']):text(10.29+j*.53,4.24,label,5.9)
text(11.65,4.24,'→',6.5)
for j,label in enumerate(['$g$','$l$','$p$']):text(12.04+j*.47,4.24,label,6.7)
for i,(row,flags) in enumerate([([0,1,1],[1,0,1]),([1,1,0],[0,1,1]),([1,1,1],[0,0,1]),([0,1,0],[1,1,0])]):
 for j,value in enumerate(row):text(10.29+j*.53,4.53+i*.29,str(value),6.5)
 for j,value in enumerate(flags):text(12.04+j*.47,4.53+i*.29,str(value),6.5)
text(11.72,5.76,'g: gain   l: loss   p: persistence',5.7)
text(11.72,6.00,'p = 1 for ≥ 2 consecutive trees',5.8)
''' + s[b:]
s=s.replace("'Factored transforms of current $u$\\n32 → 8 → 32'", "'Nonlinear factored maps of $u$\\n32 → 8 → 32'")
p.write_text(s)
print('Connected both edge-gate inputs and made overlapping clade flags explicit.')
