import os, sys
import argparse

def split_labels_file(sequence_file, output_dir):
	print ("%s, %s" % (sequence_file, output_dir))

	# TODO: get sequence name
	#sequence = "0010"

	base = os.path.basename(sequence_file)
	os.path.splitext(base)
	sequence = os.path.splitext(base)[0]

	print ("Processing sequence: %s" % sequence)

	seq_output = os.path.join(output_dir, sequence)
	if not os.path.exists(seq_output):
        	os.makedirs(seq_output)

	num_lines = 0

	# Read-in the  whole seq file
	with open(sequence_file, "r") as fin:
		# Read line
		# Parse line
		# Get current frame

		active_frame = None
		fout = None

		for line in fin:
			tokens = line.split(" ")
			parsed_frame = int(tokens[0])

			if active_frame is not None and active_frame != parsed_frame:
				fout.close()
				fout = None
				active_frame = None
				# print ("close ...")

			if active_frame == None:
				print ("Proc frame: %d" % parsed_frame)
				fnameout = os.path.join(seq_output, "%06d.txt" % parsed_frame)
				fout = open(fnameout, "w") # Open for buissness
				active_frame = parsed_frame

			#elif active_frame == parsed_frame:
			fout.write(" ".join(tokens[2:])) # Add to open file
			num_lines += 1
			# print ("write ...")
	print ("Lines: %d" % num_lines)


parser = argparse.ArgumentParser(description='Specify paths')
parser.add_argument('--output_dir', default='foobar')
parser.add_argument('--sequence_path', default='foobar')

args = parser.parse_args()

parser.print_help()

if __name__ == "__main__":
	split_labels_file(args.sequence_path, args.output_dir)
