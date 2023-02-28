import re
import csv

with open('results.txt', 'r') as infile, open('output.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Attribute', 'Protected Attribute', 'Lambda', 'Set', 'Group', 'True positives', 'True negatives',
                     'False positives', 'False negatives'])

    attribute = ''
    prot_attr = ''
    lambda_val = ''

    for line in infile:
        if line.startswith('Using attribute:'):
            attribute = line.split(':')[-1].strip()
        elif line.startswith('Protecting attribute:'):
            prot_attr = line.split(':')[-1].strip()
        elif line.startswith('Lambda value:'):
            lambda_val = line.split(':')[-1].strip()
        elif 'Evaluate on validation set' in line:
            set_type = 'validation'
            if 'for protected' in line:
                group_type = 'protected'
            else:
                group_type = 'normal'
        elif 'Evaluate on test set' in line:
            set_type = 'test'
            if 'for protected' in line:
                group_type = 'protected'
            else:
                group_type = 'normal'
        elif line.startswith('Evaluation accuracy:'):
            pass  # skip accuracy lines
        elif line.startswith('True positives:'):
            tp, tn, fp, fn = re.findall('\d+', line)
            writer.writerow([attribute, prot_attr, lambda_val, set_type, group_type, tp, tn, fp, fn])
