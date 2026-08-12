import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import warnings
import corner
import os
warnings.filterwarnings('ignore')

#ALguma coisa

mcmc_iteracoes = 5000
mcmc_burnin = 2000

def modelo_king_density(theta, r):
    sigma0, rc, rt = theta
    densidade = np.zeros_like(r, dtype=float)
    if rc >= rt:
        return densidade # Retorna zeros se os parâmetros forem inválidos
        
    termo_tidal = (1.0 + (rt / rc)**2)**(-0.5)
    mask = r < rt
    termo_r = (1.0 + (r[mask] / rc)**2)**(-0.5)
    densidade[mask] = sigma0 * (termo_r - termo_tidal)**2
    return densidade

# log-verossimilhança
def log_likelihood_poisson(theta, r, counts_obs, area_bins):
    # CORREÇÃO: Usar o nome correto da função definida
    densidade_modelo = modelo_king_density(theta, r) 
    
    lambda_esperado = densidade_modelo * area_bins
    # Evitar log(0) ou negativos
    lambda_esperado[lambda_esperado <= 0] = 1e-10 
    
    # Log-Likelihood Poisson: sum(k * ln(lambda) - lambda)
    log_L = np.sum(counts_obs * np.log(lambda_esperado) - lambda_esperado)
    return log_L

# valores aceitáveis dos parâmetros
def log_prior(theta, raio_max_plot):
    sigma0, rc, rt = theta
    
    # Restrições Físicas
    if not (0 < sigma0 < 1e7): # Limite superior generoso para densidade
        return -np.inf
    if not (0 < rc < rt): # Raio do núcleo deve ser menor que o de maré
        return -np.inf
    if not (rc < rt < raio_max_plot * 1.5): # Rt não pode ser infinitamente maior que os dados
        return -np.inf
        
    return 0.0

def log_probability(theta, r, counts_obs, area_bins, raio_max_plot):
    lp = log_prior(theta, raio_max_plot)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_poisson(theta, r, counts_obs, area_bins)


# 2. CONFIGURAÇÕES INICIAIS

pasta_dados = r'C:/Teste/DENER-sinteticos-RESULADOS'
arquivo_parametro = r'C:/Teste/DENER-sinteticos-RESULADOS/lista-testes-sinteticos+rtidal.txt'
pasta_saida = r'C:/Teste/DENER-sinteticos-RESULADOS/resultados_king'

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# Ler parâmetros sintéticos
df_parametros = pd.read_csv(arquivo_parametro, sep=';')
df_parametros.columns = df_parametros.columns.str.strip().str.replace(';', '')
df_parametros = df_parametros.apply(lambda col: col.str.strip().str.replace(';', '') if col.dtype == 'object' else col)
df_parametros = df_parametros.reset_index(drop=True)

# Lista arquivos
lista_arquivos = []
for arquivo in os.listdir(pasta_dados):
    if arquivo.endswith('.dat'):
        caminho_completo = os.path.join(pasta_dados, arquivo)
        lista_arquivos.append(caminho_completo)

# 3. LOOP DE PROCESSAMENTO

todos_resultados = []
for numero_arquivo, caminho_arquivo in enumerate(lista_arquivos, start=1):
    nome_arquivo = os.path.basename(caminho_arquivo).replace('.dat', '')
    print(f"\nProcessando: {nome_arquivo}")

    try:
        dados = pd.read_csv(caminho_arquivo, sep=r'\s+', header=0)
        membros_reais = dados[dados['Pmemb'] == 1].copy()

        c_ra = np.median(membros_reais['RA_ICRS'])
        c_dec = np.median(membros_reais['DE_ICRS'])

        r_deg = np.sqrt(
            ((membros_reais['RA_ICRS'] - c_ra) * np.cos(np.deg2rad(c_dec)))**2 +
            (membros_reais['DE_ICRS'] - c_dec)**2)

        membros_reais['r_deg'] = r_deg
        raio_max = membros_reais['r_deg'].max()
        
        raio_max_plot = raio_max * 3

        counts_membros, bin_edges = np.histogram(membros_reais['r_deg'], bins='auto')
        area_bins = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
        r_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
        density_membros = counts_membros / area_bins

        mask = counts_membros > 0
        r_fit = r_mid[mask]
        counts_fit = counts_membros[mask]
        density_fit = density_membros[mask]
        area_fit = area_bins[mask]

        counts_err = np.sqrt(counts_fit)
        counts_err[counts_err == 0] = 1
        density_err_fit = counts_err / area_fit
        print(f"Membros: {len(membros_reais)}")
        print(f"Pontos para ajuste: {len(r_fit)}")

        # Chute inicial
        chute_sigma0 = density_fit[0]
        metade_densidade = density_fit[0] / 2.0
        idx_metade = np.argmin(np.abs(density_fit - metade_densidade))
        chute_rc = r_fit[idx_metade]
        chute_rt = r_fit[-1] * 1.5

        initial_guess = [chute_sigma0, chute_rc, chute_rt]

        # MCMC
        ndim = 3
        nwalkers = 64
        p0 = initial_guess + 1e-2 * np.abs(initial_guess) * np.random.randn(nwalkers, ndim)
        p0 = np.abs(p0)
        p0[:, 2] = np.maximum(p0[:, 2], p0[:, 1] * 2.0)

        print("  Rodando MCMC...")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(r_mid, counts_membros, area_bins, raio_max_plot))
        sampler.run_mcmc(p0, mcmc_iteracoes, progress=False)

        samples = sampler.get_chain(discard=mcmc_burnin, flat=True)
        best = np.median(samples, axis=0)
        std = np.std(samples, axis=0)
        sigma0, rc, rt = best

        print(f"  σ0={sigma0:.1f} | rc={rc:.4f} | rt={rt:.4f}")

        fig_corner = corner.corner(samples, labels=["sigma0", "rc", "rt"],
                                   truths=[sigma0, rc, rt])
        fig_corner.savefig(os.path.join(pasta_saida, f'{nome_arquivo}_corner.png'))
        plt.close(fig_corner)

        # Gráficos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.errorbar(r_fit, counts_fit, yerr=counts_err, fmt='o', capsize=4, label='Dados')
        counts_modelo = modelo_king_density(best, r_fit) * area_fit
        ax1.plot(r_fit, counts_modelo, lw=2, label='Modelo King')
        ax1.axvline(rt, color='red', linestyle=':', lw=1.5, label=f'rt={rt:.4f}')
        ax1.set_xlabel('r (deg)')
        ax1.set_ylabel('Counts')
        ax1.legend()

        ax2.errorbar(r_fit, density_fit, yerr=density_err_fit, fmt='o', capsize=4, label='Dados')
        r_smooth = np.linspace(0, raio_max, 300)
        density_modelo = modelo_king_density(best, r_smooth)
        ax2.plot(r_smooth, density_modelo, lw=2, label='Modelo King')
        ax2.axvline(rt, color='red', linestyle=':', lw=1.5, label=f'rt={rt:.4f}')
        ax2.set_xlabel('r (deg)')
        ax2.set_ylabel('Densidade')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, f'{nome_arquivo}_perfil.png'), dpi=150)
        plt.close()

        # Resultado sintético
        row = df_parametros[df_parametros['name'].astype(str).str.strip() == nome_arquivo.strip()]
        if len(row) > 0:
            tidal_radius_sintetico = float(row['rtidal_arcmin'].values[0]) / 60.0
            razo = rt / tidal_radius_sintetico
            n_sigma = abs(rt - tidal_radius_sintetico) / std[2]
            print(f"  Ref: {tidal_radius_sintetico:.4f} | Razão: {razo:.4f} | Diff: {n_sigma:.2f}σ")

        else:
            tidal_radius_sintetico, razo, n_sigma = None, None, None

        todos_resultados.append({
            'arquivo': nome_arquivo, 'n_membros': len(membros_reais),
            'sigma0': sigma0, 'sigma0_err': std[0], 'rc': rc, 'rc_err': std[1],
            'rt': rt, 'rt_err': std[2],'tidal_sintetico': tidal_radius_sintetico,
            'razao': razo, 'diferenca_sigma': n_sigma})
        
    except Exception as e:
        print(f"Erro no arquivo {nome_arquivo}: {e}")

df_resultados = pd.DataFrame(todos_resultados)
df_resultados.to_csv(os.path.join(pasta_saida, 'resumo_resultados.csv'), index=False)
print("\nPROCESSAMENTO FINALIZADO")

print("Gerando gráficos de comparação...")

# Filtrar apenas linhas que tenham dados sintéticos correspondentes (não nulos)
df_plot = df_resultados.dropna(subset=['tidal_sintetico', 'rt'])

if not df_plot.empty:
    # --- GRÁFICO 1: IDENTIDADE (Rt Real vs Rt Ajustado) ---
    plt.figure(figsize=(8, 7))
    
    # Linha de identidade (y=x)
    min_val = min(df_plot['tidal_sintetico'].min(), df_plot['rt'].min())
    max_val = max(df_plot['tidal_sintetico'].max(), df_plot['rt'].max())
    # Um pequeno buffer de 5% para o gráfico não ficar colado nas bordas
    limites = [min_val * 0.9, max_val * 1.1]
    
    plt.plot(limites, limites, 'k--', alpha=0.6, label='Identidade (Ideal)')
    
    # Pontos coloridos pelo número de membros (para mostrar o efeito de N)
    sc = plt.scatter(df_plot['tidal_sintetico'], df_plot['rt'], 
                     c=df_plot['n_membros'], cmap='viridis', 
                     edgecolor='k', s=60, alpha=0.8, zorder=10)
    
    # Barras de erro (opcional, se ficar muito poluído pode comentar)
    plt.errorbar(df_plot['tidal_sintetico'], df_plot['rt'], 
                 yerr=df_plot['rt_err'], fmt='none', ecolor='gray', alpha=0.3, zorder=5)

    cbar = plt.colorbar(sc)
    cbar.set_label('Número de Membros')
    
    plt.xlabel(r'$R_t$ Real (Sintético) [deg]', fontsize=12)
    plt.ylabel(r'$R_t^{fit}$', fontsize=12)
    plt.title('Validação: Input vs Output', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal') # Mantém a proporção 1:1 visualmente
    
    caminho_identidade = os.path.join(pasta_saida, 'comparacao_Rt_identidade.png')
    plt.savefig(caminho_identidade, dpi=150)
    plt.close()
    print(f"Gráfico de identidade salvo em: {caminho_identidade}")

    # --- GRÁFICO 2: RAZÃO (VIÉS) vs NÚMERO DE MEMBROS ---
    plt.figure(figsize=(10, 6))
    
    plt.axhline(1.0, color='r', linestyle='--', label='Razão Ideal (1.0)')
    
    # Plotar Razão (Rt_fit / Rt_real)
    plt.scatter(df_plot['n_membros'], df_plot['razao'], 
                color='royalblue', edgecolor='k', s=60, alpha=0.7)
    
    plt.xlabel('Número de membros (N)', fontsize=12)
    plt.ylabel(r'Razão ($R_t^{fit} / R_t^{real}$)', fontsize=12)
    plt.title('Dependência do erro com o tamanho da amostra', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Log scale no X as vezes ajuda se tiver aglomerados muito grandes misturados com pequenos
    # plt.xscale('log') 
    
    caminho_vies = os.path.join(pasta_saida, 'analise_vies_por_membros.png')
    plt.savefig(caminho_vies, dpi=150)
    plt.close()
    print(f"Gráfico de viés salvo em: {caminho_vies}")

else:
    print("AVISO: Não foi possível gerar gráficos de comparação. Verifique se os nomes dos arquivos batem com a lista sintética.")

caminho_csv = os.path.join(pasta_saida, 'resumo_resultados.csv')

# 1. Carregar os dados
try:
    df = pd.read_csv(caminho_csv)
    # Filtra apenas linhas que têm razão calculada (tira os NaNs)
    df = df.dropna(subset=['razao'])
    
    print(f"Total de aglomerados analisados: {len(df)}")
    print("-" * 50)

    # 2. Definição de 'Perto de 1' (Margens de erro)
    # Exemplo: Margem 0.2 significa aceitar erro de 20% (Razão entre 0.8 e 1.2)
    margens = [0.10, 0.20, 0.50] 

    for margem in margens:
        # Filtra quem está dentro do limite (1 - margem) até (1 + margem)
        # Ex: para 10%, pega quem tem razão entre 0.9 e 1.1
        sucessos = df[ (df['razao'] >= (1 - margem)) & (df['razao'] <= (1 + margem)) ]
        
        qtd = len(sucessos)
        total = len(df)
        pct = (qtd / total) * 100
        
        print(f"Com margem de erro de {margem*100:.0f}% (Razão 0.{int((1-margem)*10)}-1.{int((1+margem)*10)}):")
        print(f"   -> {qtd} de {total} aglomerados ({pct:.1f}%)")

    print("-" * 50)

    # 3. A Prova Real: Comparação por número de membros
    # Vamos ver a diferença entre aglomerados POBRES (<100) e RICOS (>=100)
    # Usando margem de 20% como critério de sucesso
    margem_padrao = 0.20
    limite_inferior = 1 - margem_padrao
    limite_superior = 1 + margem_padrao

    # Grupo Pobre (N < 100)
    df_pobre = df[df['n_membros'] < 100]
    sucesso_pobre = df_pobre[ (df_pobre['razao'] >= limite_inferior) & (df_pobre['razao'] <= limite_superior) ]
    if len(df_pobre) > 0:
        pct_pobre = (len(sucesso_pobre) / len(df_pobre)) * 100
    else:
        pct_pobre = 0

    # Grupo Rico (N >= 100)
    df_rico = df[df['n_membros'] >= 100]
    sucesso_rico = df_rico[ (df_rico['razao'] >= limite_inferior) & (df_rico['razao'] <= limite_superior) ]
    if len(df_rico) > 0:
        pct_rico = (len(sucesso_rico) / len(df_rico)) * 100
    else:
        pct_rico = 0

    print(f"COMPARATIVO DE QUALIDADE (Margem de 20%):")
    print(f"Aglomerados Pobres (N < 100): {pct_pobre:.1f}% de sucesso.")
    print(f"Aglomerados Ricos  (N >= 100): {pct_rico:.1f}% de sucesso.")
    print("-" * 50)

except FileNotFoundError:
    print("Erro: O arquivo 'resumo_resultados.csv' não foi encontrado na pasta de saída.")
except Exception as e:
    print(f"Erro ao processar estatísticas: {e}")








