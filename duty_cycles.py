import numpy as np

# x, y, z represent LIGO H only, LIGO L only and Virgo only times
# a, b, c represent (LIGO H + LIGO L), (LIGO H + Virgo), (LIGO L + Virgo) times
# r represents the time when all three were on

#a,b,c,x,y,z,r
A = np.array([[1,1,1,1,1,1,1],
              [0,0,0,0,0,0,1],
              [1,1,0,1,0,0,1],
              [1,0,1,0,0,1,1],
              [0,1,1,0,1,0,1],
              [1,1,1,0,0,0,0],
              [0,0,0,1,1,1,0],
              ])

# Information from https://gwosc.org/detector_status/O3a/
B = np.array([96.8, 44.5, 71.2, 75.8, 76.3, 37.4, 15])

vals = np.linalg.pinv(A) @ B.T
# vals = np.linalg.solve(A,B.T)

print(vals)
print(sum(vals))

a,b,c,x,y,z,r = vals

corr = np.array([[100, a + r, b + r],
                [a + r, 100, c + r],
                [b + r, c + r, 100]])

print(corr)

# from matplotlib_venn import venn3, venn3_circles
# from matplotlib import pyplot as plt
  
# # depict venn diagram
# venn3(subsets=vals, 
#       set_labels=('LIGO H', 'LIGO L', 'Virgo'), 
#       set_colors=("orange", "blue", "red"), alpha=0.7)
  
# # outline of circle line style and width
# venn3_circles(subsets=vals,
#               linestyle="dashed", linewidth=2)
  
# # title of the venn diagram
# plt.title("Venn Diagram in geeks for geeks")
# plt.show()