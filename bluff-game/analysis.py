import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

def get_win_rates(filename):
    df = pd.read_csv(filename)
    agent0_mean = df['Agent 0 Win Rate'].mean()
    agent0_se = stats.sem(df['Agent 0 Win Rate'])
    agent1_mean = df['Agent 1 Win Rate'].mean()
    agent1_se = stats.sem(df['Agent 1 Win Rate'])
    draw_rate = df['Draw Rate'].mean()
    draw_rate_se = stats.sem(df['Draw Rate'])
    return (agent0_mean, agent0_se), (agent1_mean, agent1_se), (draw_rate, draw_rate_se)


tom0_tom0_rates = get_win_rates('zero_zero.csv')
tom0_tom1_rates = get_win_rates('zero_first.csv')
tom0_tom2_rates = get_win_rates('zero_second.csv')
tom1_tom1_rates = get_win_rates('first_first.csv')
tom1_tom2_rates = get_win_rates('first_second3.csv')
tom2_tom2_rates = get_win_rates('second_second.csv')


agent1_rates = [tom0_tom0_rates[0][0], tom0_tom1_rates[0][0], tom0_tom2_rates[0][0], 
                tom1_tom1_rates[0][0], tom1_tom2_rates[0][0], tom2_tom2_rates[0][0]]
agent1_ses = [tom0_tom0_rates[0][1], tom0_tom1_rates[0][1], tom0_tom2_rates[0][1], 
              tom1_tom1_rates[0][1], tom1_tom2_rates[0][1], tom2_tom2_rates[0][1]]

agent2_rates = [tom0_tom0_rates[1][0], tom0_tom1_rates[1][0], tom0_tom2_rates[1][0], 
                tom1_tom1_rates[1][0], tom1_tom2_rates[1][0], tom2_tom2_rates[1][0]]
agent2_ses = [tom0_tom0_rates[1][1], tom0_tom1_rates[1][1], tom0_tom2_rates[1][1], 
              tom1_tom1_rates[1][1], tom1_tom2_rates[1][1], tom2_tom2_rates[1][1]]

plt.figure(figsize=(14, 7))

width = 0.35
x = np.arange(len(agent1_rates))


colors_agent1 = ['lightskyblue', 'lightskyblue', 'lightskyblue', 
                 'lightgreen', 'lightgreen', 'lightcoral']
colors_agent2 = ['lightskyblue', 'lightgreen', 'lightcoral', 
                 'lightgreen', 'lightcoral', 'lightcoral']

plt.grid(axis='y', linestyle='--', alpha=0.7)

bars1 = plt.bar(x - width/2, agent1_rates, width, label='Agent 1', 
                color=colors_agent1, yerr=agent1_ses, 
                capsize=5, error_kw={'ecolor': '0.3', 'capthick': 1.5})
bars2 = plt.bar(x + width/2, agent2_rates, width, label='Agent 2', 
                color=colors_agent2, yerr=agent2_ses, 
                capsize=5, error_kw={'ecolor': '0.3', 'capthick': 1.5})


plt.ylabel('Win Rate', fontdict={'fontsize': 16})
plt.yticks(fontsize=14) 
plt.title('Theory of Mind Agent Win Rates with Standard Error', fontdict={'fontsize': 16})
plt.xticks(x, ['ToM0 vs\nToM0', 'ToM0 vs\nToM1', 'ToM0 vs\nToM2', 
               'ToM1 vs\nToM1', 'ToM1 vs\nToM2', 'ToM2 vs\nToM2'], 
           rotation=0, fontsize=14)


from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightskyblue', label='ToM0'),
                   Patch(facecolor='lightgreen', label='ToM1'),
                   Patch(facecolor='lightcoral', label='ToM2')]
plt.legend(handles=legend_elements, title='Theory of Mind Level', loc='best', fontsize=14)


plt.tight_layout()
plt.show()