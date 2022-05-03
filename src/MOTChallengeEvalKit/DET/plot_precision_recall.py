import os, sys
import matplotlib.pyplot as plt
import numpy as np

# Import globals
from GLOBALS import globals

# Get output path
output_dir = os.path.join(globals.bmttDir, 'pub', 'plots')

def plot_pr(shakey, plot_variables, challenge = None, short_name = None,  grid=True):
	rc, pr = plot_variables
	print( rc)
	print(pr)
	if len(rc) != len(pr):
		raise Exception("Error, pr/rc array length does not match!")

	# Plot
	plt.figure()
	plt.plot(rc, pr, linewidth=3.0, color="blue")
	ax = plt.gca()

	plt.xlabel("Recall")
	plt.ylabel("Precision")

	if not challenge or not short_name:
		# Set title, axes names
		plt.title("ROC Curve")
	else:
		plt.title("ROC Curve - %s - %s" %(challenge, short_name))

	# Set scales
	ax.set_ylim([0.0, 1.0])
	ax.set_xlim([0.0, 1.0])
	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.set_xticks(np.arange(0, 1.1, 0.1))

	if grid == True:
		plt.grid()

	# Save
	dir = os.path.join( output_dir, shakey)
	if not os.path.exists(dir):
            os.makedirs(dir)
	output_path = os.path.join(dir,  "rcpr-test.png")
	print ("Saving to: %s"%output_path)
	plt.savefig(output_path, bbox_inches='tight')
