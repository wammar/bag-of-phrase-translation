import argparse
import utils

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-ref')
parser.add_argument('-tgt')
opts = parser.parse_args()

print utils.bleu(opts.ref, opts.tgt)

