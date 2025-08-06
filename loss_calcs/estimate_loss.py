# according to Ainley and Mathieson
# 

import json


def make_new_config(fname='default.json'):
    t = float(input(f'what is the blade thickness'))
    c = float(input(f'What is the blades chord'))

    # beta2 = input angle between relative and axial
    # beta3 = outlet angle between relative and axial
    print(f'input the Y_p coefficient from cascasde data, based on t/c = {t/c:.2f}\n'
                f'See this link: https://i.imgur.com/DVtoGcX.png')

    Y_p_nozz = float(input(f'input the value for the full nozzle curve'))
    Y_p_reaction = float(input(f'input the value for the full reaction blade curve'))
    # If nozzle type, then b2 = alpha1; beta3 = alpha2

    beta_2 = float(input(f'What is beta_2'))
    beta_3 = float(input(f'What is beta_3'))

    j = {'t/c':t/c,
        'Y_p_nozz':Y_p_nozz,
         'Y_p_reaction':Y_p_reaction,
         'beta_2': beta_2,
         'beta_3': beta_3}
    
    with open(fname, 'w') as f:
        json.dump(j, f, indent=4)


def loss(config_dict):
    with open(config_dict, 'r') as f:
        j = json.load(f)


    Y_p = (j['Y_p_nozz'] + (j['beta_2']/j['beta_3'])**2 * (j['Y_p_nozz'] - j['Y_p_reaction'])) * (5 * j['t/c'])**(j['beta_2']/j['beta_3']) # y_p interpolation
    print(f'Y_p = {Y_p}')

if __name__ == "__main__":
    fname = 'test.json'
    make_new_config(fname) # run if is the first time
    loss(fname)