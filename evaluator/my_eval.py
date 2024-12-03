import os

teratum_result_path = '../result/teratum_result'
pred_result_path = '../result/pred_result'

def my_eval(cat, name, re_exp= False):
    # 저장 목표
    dst = rf'../result/NmAP50_result/{cat}/{name}.txt'

    # 중복 검사
    if((re_exp == False) & (os.path.exists(dst))):
        print(f'{cat} - {name} is already exist, 스킵!!!')
        
        return
    elif((re_exp == True) & (os.path.exists(dst))):
        print(f'{dst} is already exist, but 다시!!!')

    os.system(f'python evaluate.py --category {cat} --name {name}')
    print(f'{cat} - {name} is done')

def eval_by_dir(dir_name, re_exp= False):
    name_list = os.listdir(f'{teratum_result_path}/{dir_name}')

    for exp_name in name_list:
        if(exp_name[-4:] == '.tar'):
            # print(dir_name, exp_name[:-4])
            my_eval(dir_name, exp_name[:-4], re_exp= re_exp)

def eval_allll(re_exp= False):
    category_list = os.listdir(teratum_result_path)

    for category in category_list:
        if(category == 'undefined'):
            continue
        elif(category == '.gitkeep'):
            continue
        else:
            eval_by_dir(category, re_exp= re_exp)

def my_eval_gpu(cat, name, re_exp= False):
    # 저장 목표
    dst = rf'../result/GmAP50_result/{cat}/{name}.txt'

    # 중복 검사
    if((re_exp == False) & (os.path.exists(dst))):
        print(f'{cat} - {name} is already exist, 스킵!!!')
        
        return
    elif((re_exp == True) & (os.path.exists(dst))):
        print(f'{dst} is already exist, but 다시!!!')

    os.system(f'python evaluate_gpu.py --category {cat} --name {name}')
    print(f'{cat} - {name} is done')

def eval_by_dir_gpu(dir_name, re_exp= False):
    name_list = os.listdir(f'{pred_result_path}/{dir_name}')

    for exp_name in name_list:
        if(exp_name[-4:] == '.tar'):
            # print(dir_name, exp_name[:-4])
            my_eval_gpu(dir_name, exp_name[:-4], re_exp= re_exp)

def eval_allll_gpu(re_exp= False):
    category_list = os.listdir(pred_result_path)

    for category in category_list:
        if(category == 'undefined'):
            continue
        elif(category == '.gitkeep'):
            continue
        else:
            eval_by_dir_gpu(category, re_exp= re_exp)