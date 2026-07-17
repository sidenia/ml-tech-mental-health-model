# Dicionário de Dados

Fonte: OSMI Mental Health in Tech Survey 2014.
Colunas principais usadas no modelo (lista completa no CSV bruto tem 27 colunas; as demais são descartadas em `src/data.py` — registrar motivo ao descartar).

| Coluna | Tipo | Valores | Transformação | Descrição |
|---|---|---|---|---|
| `Age` | int | 18–75 (após limpeza) | filtro de faixa + StandardScaler | idade do respondente |
| `Gender` | str livre | → `male` / `female` / `other` | normalização manual + OneHot | gênero (texto livre no bruto, ~49 variações) |
| `family_history` | str | yes / no | OneHot | histórico familiar de doença mental |
| `work_interfere` | str | never / rarely / sometimes / often / NaN | NaN → "unknown", OneHot | condição mental interfere no trabalho? |
| `self_employed` | str | yes / no / NaN | NaN → "no", OneHot | autônomo? |
| `remote_work` | str | yes / no | OneHot | trabalha remoto ≥50% do tempo? |
| `tech_company` | str | yes / no | OneHot | empregador é empresa de tecnologia? |
| `benefits` | str | yes / no / don't know | OneHot | empregador oferece benefícios de saúde mental? |
| `care_options` | str | yes / no / not sure | OneHot | conhece opções de cuidado oferecidas? |
| `no_employees` | str | faixas (1-5 … 1000+) | OrdinalEncoder | tamanho da empresa |
| **`treatment`** | str | yes / no | **ALVO** → binário 1/0 | buscou tratamento de saúde mental? |

## Colunas descartadas (e por quê)

| Coluna | Motivo |
|---|---|
| `Timestamp` | data da resposta, sem valor preditivo |
| `comments` | texto livre, 87% vazio (1.095 nulos de 1.259) |
| `state` | 41% nulo (515), só se aplica a respondentes dos EUA |
| `Country` | 60% EUA; dezenas de categorias com poucos exemplos cada (risco de overfitting) |
| `anonymity`, `leave`, `coworkers`, `supervisor`, `mental_health_consequence`, `phys_health_consequence`, `mental_health_interview`, `phys_health_interview`, `mental_vs_physical`, `obs_consequence`, `seek_help`, `wellness_program` | políticas/percepções da empresa — cortadas para manter escopo didático; candidatas a experimento futuro (reincluir e medir F1) |
