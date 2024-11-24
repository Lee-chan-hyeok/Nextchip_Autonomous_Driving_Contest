import os

teratum_result_path = '../result/teratum_result'

def my_eval(cat, name, re_exp= False):
    # 저장 목표
    dst = rf'../result/{cat}/{name}.txt'

    # 중복 검사
    if((re_exp == False) & (os.path.exists(dst))):
        print(f'{dst} is already exist, 스킵!!!')
        
        return
    elif((re_exp == True) & (os.path.exists(dst))):
        print(f'{dst} is already exist, but 다시!!!')

    os.system(f'python evaluate.py --category {cat} --name {name}')

def eval_by_dir(dir_name, re_exp= False):
    name_list = os.listdir(f'{teratum_result_path}/{dir_name}')

    for exp_name in name_list:
        if(exp_name[-4:] == '.txt'):
            print(dir_name, exp_name[:-4])
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