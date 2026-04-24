import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statistics as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
#==================================================================================================#
# 1. CARREGAMENTO E DEFINIÇÕES
caminho_arquivo = "diabetes.csv"
df = pd.read_csv(caminho_arquivo)
colunas_fisiologicas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 2. ANÁLISE DOS DADOS ORIGINAIS (SUJOS)
stats_original = df.describe()
contagem_zeros = (df[colunas_fisiologicas] == 0).sum()
dist_original = df['Outcome'].value_counts()

# 3. LIMPEZA DOS DADOS
# Substituindo zeros por NaN e removendo as linhas
df_processado = df.copy()
df_processado[colunas_fisiologicas] = df_processado[colunas_fisiologicas].replace(0, np.nan)
df_limpo = df_processado.dropna()

# 4. ANÁLISE DOS DADOS LIMPOS
stats_limpo = df_limpo.describe()
dist_limpo = df_limpo['Outcome'].value_counts()
medias_por_diagnostico = df_limpo.groupby('Outcome').mean()
skewness_limpo = df_limpo.drop('Outcome', axis=1).skew()
# Lista de colunas que você quer analisar
colunas_de_interesse = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

#==================================================================================================#
# ---------------------------------------------------------
# 1. MÉTRICAS AVANÇADAS VIA PANDAS
# ---------------------------------------------------------
resumo_estatistico = df_limpo.drop('Outcome', axis=1).agg(['mean', 'median', 'skew', 'kurt']).T
resumo_estatistico.columns = ['Média', 'Mediana', 'Assimetria (Skew)', 'Curtose (Excesso)']

display(resumo_estatistico)

#==================================================================================================#
#4.2 Teste de Aderência Matemática
colunas_para_testar = df_limpo.columns.drop('Outcome')

distribuicoes = {
    'normal': stats.norm,
    'lognormal': stats.lognorm,
    'gamma': stats.gamma,
    'weibull_min': stats.weibull_min
}

resultados_completos = []

for col in colunas_para_testar:
    dados = df_limpo[col].dropna()

    # Tratamento de segurança para distribuições que não aceitam zero exato
    if dados.min() <= 0:
        dados_fit = dados + 0.0001
    else:
        dados_fit = dados

    # Loop que testa CADA modelo e salva TODOS os resultados
    for nome, dist in distribuicoes.items():
        try:
            params = dist.fit(dados_fit)
            ks_stat, ks_pvalor = stats.kstest(dados_fit, dist.cdf, args=params)

            # Cálculo do AIC
            loglik = np.sum(dist.logpdf(dados_fit, *params))
            k = len(params)
            aic = 2 * k - 2 * loglik

            resultados_completos.append({
                'variavel': col,
                'distribuicao': nome,
                'AIC': round(aic, 4),
                'KS_stat': round(ks_stat, 4),
                'KS_pvalor': round(ks_pvalor, 4)
            })
        except:
            continue # Se divergir, ignora

# Cria o DataFrame e ordena por Variável e depois pelo AIC
tabela_aderencia = pd.DataFrame(resultados_completos)
tabela_aderencia = tabela_aderencia.sort_values(by=['variavel', 'AIC']).reset_index(drop=True)

display(tabela_aderencia)
#==================================================================================================#

#4.3 Os melhores ajustes de distribuições
# ---------------------------------------------------------
# 3. TABELA RESUMO: A MELHOR DISTRIBUIÇÃO POR VARIÁVEL
# ---------------------------------------------------------
# Agrupa pela variável e pega o índice da linha que tem o menor AIC
indices_campeoes = tabela_aderencia.groupby('variavel')['AIC'].idxmin()

# Filtra a tabela original
tabela_campeoes = tabela_aderencia.loc[indices_campeoes].reset_index(drop=True)
tabela_campeoes = tabela_campeoes.rename(columns={'distribuicao': 'melhor_distribuicao'})


display(tabela_campeoes)

#==================================================================================================#
#4.4 Verificação Visual e Conclusão para Modelagem
# ---------------------------------------------------------
# 4. VERIFICAÇÃO VISUAL DA MAIS ASSIMÉTRICA (Insulina)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

dados_insulina = df_limpo['Insulin'].dropna()

# Descobre automaticamente a melhor distribuição na tabela
melhor_dist_insulina = tabela_campeoes[tabela_campeoes['variavel'] == 'Insulin'].iloc[0]['melhor_distribuicao']

# 1. Histograma base
dados_insulina.plot.hist(density=True, bins=30, color='royalblue', alpha=0.3, label='Frequência dos Dados (Real)')

x = np.linspace(dados_insulina.min(), dados_insulina.max(), 200)

# 2. Plota a curva teórica VENCEDORA (Vermelha)
dist_vencedora = distribuicoes[melhor_dist_insulina]
params_venc = dist_vencedora.fit(dados_insulina if dados_insulina.min() > 0 else dados_insulina + 0.0001)
plt.plot(x, dist_vencedora.pdf(x, *params_venc), color='red', lw=3, label=f'Ajuste lognormal ({melhor_dist_insulina.capitalize()})')

# 3. Plota a curva NORMAL para contraste (Preta Tracejada)
params_norm = stats.norm.fit(dados_insulina)
plt.plot(x, stats.norm.pdf(x, *params_norm), color='black', lw=2, linestyle='--', label='Ajuste normal')

plt.title('Insulina comparação entre ajustes', fontsize=16, fontweight='bold')
plt.xlabel('Insulina (mu U/ml)', fontsize=12)
plt.ylabel('Densidade de Probabilidade', fontsize=12)
plt.legend()
plt.xlim(dados_insulina.min(), dados_insulina.max())
plt.show()
#==================================================================================================#
#5.1 Comparação do Perfil Fisiológico
# 1. Cálculo das médias
medias_por_diagnostico = df_limpo.groupby('Outcome').mean()

display(medias_por_diagnostico.T)

# 2. Verificação Visual: Painel de Múltiplos Eixos (Subplots)
sns.set_theme(style="whitegrid")

# Criando uma grade 2x4 para abrigar as 8 variáveis separadamente
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
fig.suptitle('Comparação de Médias Fisiológicas: Saudáveis (0) vs Diabéticos (1)', fontsize=22, fontweight='bold')

colunas = medias_por_diagnostico.columns
cores = ['#4169E1', '#DC143C'] # Azul (0) e Vermelho (1)

# Loop para desenhar cada mini-gráfico
for i, col in enumerate(colunas):
    row = i // 4
    col_idx = i % 4
    ax = axes[row, col_idx]

    # Plotando a barra da variável específica
    medias_por_diagnostico[col].plot(kind='bar', ax=ax, color=cores, edgecolor='black', alpha=0.85)

    # Ajustes estéticos de cada mini-gráfico
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Saudável (0)', 'Diabético (1)'], rotation=0, fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Valor Médio' if col_idx == 0 else '') # Bota legenda no Y só na primeira coluna

# Ajuste fino para os gráficos não se atropelarem
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#==================================================================================================#
#5.2 Matriz de correlação de pearson
# Selecionando os fatores solicitados + Resultado (Outcome)
fatores = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Outcome']
df_mapa = df_limpo[fatores]

# Configurando o tamanho e o estilo da figura
plt.figure(figsize=(10, 8))
sns.set_theme(style="white") # Fundo branco para destacar as cores do heatmap

# Gerando a Matriz de Pearson
# Parâmetros adicionados: vmin e vmax travam a escala de cores de -1 a 1. fmt='.2f' formata as casas decimais.
sns.heatmap(df_mapa.corr(), annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)

plt.title('Influência dos Fatores Biológicos na Diabetes (Pearson)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.show()

#==================================================================================================#
#5.3 Mapa de Calor
# 1. Preparação e Agrupamento
df_evolucao = df_limpo.copy()
df_evolucao['Glucose_Bins'] = pd.cut(df_evolucao['Glucose'], bins=10)
df_tendencia = df_evolucao.groupby('Glucose_Bins', observed=True).mean().drop(columns=['Glucose'])

# 2. Normalização
scaler = MinMaxScaler()
df_tendencia_norm = pd.DataFrame(
    scaler.fit_transform(df_tendencia),
    columns=df_tendencia.columns,
    index=[f"Nível {i+1}" for i in range(10)]
).T

# 3. Configuração da Figura
plt.figure(figsize=(14, 7))
sns.set_theme(style="white")

# 4. Geração do Mapa de Calor (Escala de Alerta)
# cmap='YlOrRd': Amarelado (baixo) -> Alaranjado (médio) -> Avermelhado (alto)
heatmap_alert = sns.heatmap(
    df_tendencia_norm,
    annot=False,
    cmap='YlOrRd',         # Paleta de cores "quentes" (vermelho, no caso)
    linewidths=1.5,
    linecolor='white',
    cbar_kws={'label': 'Intensidade: Baixa (Amarelo) → Alta (Vermelho)'}
)

# 5. Ajustes de Texto
plt.title('Mapa de Calor de Risco: Progressão de Indicadores vs. Glicose',
          fontsize=18, fontweight='bold', pad=25)
plt.xlabel('Níveis Crescentes de Glicose', fontsize=13)
plt.ylabel('Variáveis Analisadas', fontsize=13)

# 6. Finalização
plt.tight_layout()
plt.show()
#==================================================================================================#
#5.4 O efeito de um valor extremo analizados no grafico de boxplot


#Lucas Pereira
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

# Lista de colunas que você quer analisar
colunas_de_interesse = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Definindo o estilo fora para não repetir código
estilo = dict(
    tick_labels=['Original', 'Limpo'],
    patch_artist=True, widths=0.5,# permite que você pinte o interior da caixa (usando o facecolor abaixo).
    boxprops=dict(facecolor='#bdd7e7', edgecolor='#08519c', linewidth=1.5),#A Caixa - Representa os 50% centrais dos dados)
    medianprops=dict(color='#cb181d', linewidth=2.2),#linha vermelha
    whiskerprops=dict(color='#08519c', linewidth=1.2),#As Hastes - Dispersão dos dados "normais")
    capprops=dict(color='#08519c', linewidth=1.2),#As barrinhas nas pontas das hastes
    flierprops=dict(marker='o', markerfacecolor='#cb181d', markeredgecolor='#7f0000', markersize=6),#pontos vermelhos
)

# --- A função do professor

def resumo_numerico(dados):
    dados_ordenados = sorted(dados)#poe em ordem crescente ou decrescente
    q1, q2, q3 = np.percentile(dados_ordenados, [25, 50, 75])#medida estatística que indica o valor abaixo do qual uma determinada porcentagem de observações em um conjunto de dados cai.
    #percentile define o q1,q2 e o q3
    return {
        'media': st.mean(dados), #soma dos valores divido pelo total
        'mediana': st.median(dados),#é a metade exata do numero de valores
        'q1': q1, 
        'q3': q3,
        'desvio_padrao': st.stdev(dados),#indica o quanto os dados de um conjunto estão próximos ou distantes da média
        'amplitude': max(dados) - min(dados),
        'iiq': q3 - q1,#são os valores dentro da box
    }
# Loop para processar e plotar cada coluna
for col in colunas_de_interesse:
    #vai pegar cada coluna de valor e rodar ela
    # 1. Preparação dos dados
    dados_orig = df[col] #dados sujos
    dados_limpos = df_limpo[col]#dados limpos

    # 2. Cálculo das médias
    medias = [np.mean(dados_orig), np.mean(dados_limpos)] #media de cada dado

    # 3. Plotagem
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), gridspec_kw={'width_ratios': [1.0, 1.15]})
    #subplots 1,2 = cria uma linha e duas colunas plots,como se fosse uma matriz
    #width_ratios: Isso define o tamanho lateral dos graficos.Esqueda 1 e direita 1.15
    # Gráfico 1: Visão completa
    #o axes 0 indica a coluna 0 ou a primeira coluna
    axes[0].boxplot([dados_orig, dados_limpos], **estilo)
    axes[0].grid(axis='y', linestyle='-', alpha=0.9)
    axes[0].grid(axis='x', linestyle='-', alpha=1.0)
    axes[0].scatter([1, 2], medias, color='#238b45', s=65, zorder=3, label='Média')# scatter é a linha que pinta o ponto verde no seu gráfico.zorder é a camada
    axes[0].set_title(f'Visão completa: {col}', pad=10)#pad 10 é o espaçamento do texto para a linha do topo do grafico
    axes[0].set_ylabel('Valor da variável')
    axes[0].legend(frameon=False, loc='upper right')

    # Gráfico 2: Zoom na faixa central
    axes[1].boxplot([dados_orig, dados_limpos], showfliers=False, **{k: v for k, v in estilo.items() if k != 'flierprops'})
    #showfliers false não mostra as bolinhas vermelhas. k:...vai copiar tudo, exceto a configuração chamada flierprops
    axes[1].scatter([1, 2], medias, color='#238b45', s=65, zorder=3)
    axes[1].set_title(f'Zoom na faixa central: {col}', pad=10)
    # axes[1].grid(alpha=0.35)
    axes[1].grid(axis='x', linestyle='-', alpha=0.8)
    axes[1].grid(axis='y', linestyle='-', alpha=0.8)

    plt.tight_layout()#reorganiza os dados dos grafico para que fique tudo alinhado,sem sobreposição.
    plt.show()

    # Imprimir resumo numérico para cada coluna
    print(f'--- Estatísticas de {col} ---')
    print('Original:', resumo_numerico(dados_orig))
    print('Limpo:', resumo_numerico(dados_limpos))
    print('\n')
#==================================================================================================#
#5.5 Scatter Plot: Glicose vs IMC
# Configurando o estilo
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")

# Criando o Gráfico de Dispersão (Scatter Plot)
# Usamos alpha=0.7 para que pontos sobrepostos fiquem mais escuros (densidade)
sns.scatterplot(
    x='Glucose',
    y='BMI',
    hue='Outcome',
    palette=['#4169E1', '#DC143C'], # Azul para 0, Vermelho para 1
    data=df_limpo,
    s=70,          # Tamanho dos pontos
    alpha=0.7,     # Transparência
    edgecolor='k'  # Borda preta nos pontos para contraste
)

# Adicionando linhas de referência (Médias de cada grupo para Glicose)
media_glicose_saudavel = df_limpo[df_limpo['Outcome'] == 0]['Glucose'].mean()
media_glicose_diabetico = df_limpo[df_limpo['Outcome'] == 1]['Glucose'].mean()

plt.axvline(media_glicose_saudavel, color='#4169E1', linestyle='--', alpha=0.5, label='Média Glicose (Saudável)')
plt.axvline(media_glicose_diabetico, color='#DC143C', linestyle='--', alpha=0.5, label='Média Glicose (Diabético)')

plt.title('Dispersão Fisiológica: Glicose vs IMC', fontsize=16, fontweight='bold')
plt.xlabel('Glicose (mg/dL)', fontsize=12)
plt.ylabel('Índice de Massa Corporal - IMC', fontsize=12)

# Ajustando a legenda
plt.legend(title='Diagnóstico (Outcome)', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

#==================================================================================================#













