import json

def preprocess(input_path, output_path):
    toefl_data = json.load(open(input_path))
    f_output = open(output_path, "wt")

    for entity in toefl_data:
        for sent in entity['user_essay']:
            f_output.write(sent + "\n")
    
def postprocess(ori_input, input_path, output_path):
    toefl_data = json.load(open(ori_input))
    f_output = open(input_path, "rt")


    idx = 0
    cnt = 0
    for line in f_output:
        line = line.strip()
        
        if 'gec_essay' in toefl_data[idx]:
            toefl_data[idx]['gec_essay'].append(line)
        else:
            toefl_data[idx]['gec_essay'] = [line]
        cnt += 1


        if cnt == len(toefl_data[idx]['user_essay']):
            cnt = 0
            idx += 1
    
    json.dump(toefl_data, open(output_path))







if __name__ == '__main__':
