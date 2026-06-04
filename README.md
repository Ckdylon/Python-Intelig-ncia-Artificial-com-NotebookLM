# 🧠 Aprendendo Python & Inteligência Artificial com NotebookLM

## 📖 Contexto e Objetivos
Este projeto foi desenvolvido como parte de um desafio prático na **DIO (Digital Innovation One)**. O objetivo principal é construir um **Caderno Temático Inteligente** utilizando o **Google NotebookLM** para acelerar, organizar e consolidar o aprendizado da linguagem **Python** aplicada ao ecossistema de **Inteligência Artificial (IA)** e **Machine Learning (ML)**.

**Objetivos de Estudo:**
* Compreender a sintaxe de Python aplicada à análise, limpeza e manipulação de dados.
* Dominar o ecossistema de bibliotecas para Machine Learning clássico e Deep Learning.
* Compreender a arquitetura de sistemas modernos de IA Generativa, especificamente a engenharia de sistemas RAG (*Retrieval-Augmented Generation*).
* Praticar Engenharia de Prompts para sumarização e extração de conhecimento técnico.

---

## 📚 Curadoria de Fontes
O NotebookLM foi alimentado com uma seleção de materiais didáticos e documentações técnicas focadas no desenvolvimento em Python para IA:

1. **Fundamentos de Python e POO:** Artigos e tutoriais focados em lógica de programação, sintaxe limpa e desenvolvimento orientado a objetos.
2. **Documentação do Ecossistema Data Science:** Guias práticos sobre a pilha científica (*SciPy Stack*) e algoritmos de aprendizado supervisionado e não supervisionado.
3. **Guia de Arquiteturas GenAI e Sistemas RAG:** Tutoriais práticos demonstrando a construção de agentes inteligentes utilizando *LangChain* e APIs da *OpenAI* para consulta de bases de dados em PDF.

---

## 🔬 Engenharia de Prompts e "Cicatrizes" (Troubleshooting)
Abaixo está o registro dos testes de prompts realizados para extrair insights estruturados do material sintetizado, documentando o processo de refino e os desafios encontrados.

### 🔹 Teste 1: Casos de Uso do Python em IA
* **Prompt Inicial (Ingênuo):** *"Como o Python é utilizado em Machine Learning e IA?"*
* **Resultado Obtido:** A IA gerou uma resposta abrangente e excelente, dividindo o uso do Python em 4 frentes: tratamento de dados, fluxo tradicional de ML, IA Generativa (RAG) e implantação de interfaces (*Streamlit*).
* **Lição Aprendida:** Prompts abertos funcionam bem no NotebookLM para mapear o escopo geral do documento antes de aprofundar em tópicos específicos.

### 🔹 Teste 2: Mapeamento do Ecossistema de Bibliotecas
* **Prompt Inicial:** *"Quais são as principais bibliotecas Python para Machine Learning hoje?"*
* **Resultado Obtido:** Uma categorização técnica impecável segregando as ferramentas por especialidade (Machine Learning clássico, Deep Learning, Inferência Estatística, Manipulação de Base e Visualização).
* **Troubleshooting (Cicatrizes):** Inicialmente, a IA tendeu a misturar ferramentas legadas (como *Theano*) com as líderes de mercado. Foi necessário instruir o prompt a focar no **ecossistema recente de IA Generativa**, o que forçou a inclusão do framework *LangChain* e bancos vetoriais como *ChromaDB* para sistemas RAG.

---

## 🚀 Miniguia de Estudo (Entrega Final)

### 📌 Resumos Estruturados e Fluxos de Trabalho

#### 1. O Ciclo de Vida de Machine Learning em Python
O desenvolvimento de um modelo tradicional de Machine Learning dentro do ecossistema Python segue um fluxo de trabalho estritamente estruturado e sequencial:

1. **Pré-processamento de Dados:** Limpeza, padronização e tratamento de variáveis. Utiliza-se `Pandas` para manipulação de DataFrames e classes como `StandardScaler` do `Scikit-learn` para colocar os recursos na mesma escala numérica.
2. **Divisão dos Dados:** Separação da base de dados através da função `train_test_split()`, comumente alocando **70%** dos dados para treinamento (ajuste do modelo) e **30%** para testes (validação com dados não vistos).
3. **Treinamento e Predição:** Inicialização do algoritmo e ajuste através do método `.fit(X, Y)`. Uma vez ajustado, o modelo realiza predições em novos cenários através do método `.predict(dados)`.
4. **Avaliação de Desempenho:** Geração de relatórios de métricas (como precisão, exatidão e recall) utilizando módulos como o `classification_report` do `Scikit-learn`.

#### 2. Mapeamento de Ferramentas por Especialidade
O ecossistema Python para IA é modular. A tabela abaixo correlaciona as principais bibliotecas mapeadas e suas respectivas aplicações de mercado:

| Categoria | Biblioteca(s) | Aplicação Principal no Mercado |
| :--- | :--- | :--- |
| **Computação Base** | `NumPy` | Processamento matricial e vetorial massivo e ultrarrápido (`ndarray`). |
| **Manipulação de Dados** | `Pandas` | Exploração, limpeza e transformação de dados tabulares estruturados. |
| **Matemática Avançada** | `SciPy` | Integração numérica, álgebra linear de ponta e otimização de funções. |
| **ML Clássico** | `Scikit-learn` | Modelos supervisionados e não supervisionados (Regressão, SVM, Árvores). |
| **Deep Learning** | `PyTorch` / `TensorFlow` | Redes neurais complexas otimizadas para execução paralela em GPUs. |
| **Inferência Estatística** | `Statsmodels` | Testes de hipóteses, análise de valores-p e compreensão de variáveis. |
| **Visualização** | `Matplotlib` / `Seaborn` | Plotagem de gráficos estatísticos em 2D para análise exploratória. |
| **IA Generativa / RAG** | `LangChain` | Orquestração de agentes, integração de LLMs e bancos vetoriais. |
| **Deploy / Produção** | `Streamlit` | Criação rápida de interfaces web interativas sem necessidade de HTML/CSS. |

#### 3. Entendendo a Arquitetura RAG (*Retrieval-Augmented Generation*)
Para solucionar o problema de "alucinação" das LLMs e permitir que elas consultem dados privados (como arquivos PDF), o Python é utilizado como a camada de integração (código de colagem) de sistemas RAG:
* **Fase 1 (Ingestão):** Documentos de texto são fragmentados em pequenos pedaços (*chunks*).
* **Fase 2 (Vetorização):** Esses pedaços são convertidos em representações numéricas (*embeddings*) e salvos em bancos de dados vetoriais como o `ChromaDB`.
* **Fase 3 (Busca Semântica):** Quando o usuário faz uma pergunta, o sistema a transforma em vetor, localiza o contexto mais similar no banco e envia à LLM (como a API da OpenAI) apenas a informação estritamente relevante para formular a resposta.

---

### 📕 Glossário Técnico de Conceitos
* **Python:** Linguagem de programação de alto nível, interpretada, cuja simplicidade sintática a tornou o padrão global para IA.
* **LLM (Large Language Model):** Modelos massivos de linguagem baseados em redes neurais treinados para compreender e gerar texto (ex: GPT, Gemini).
* **RAG (Retrieval-Augmented Generation):** Técnica que estende a capacidade de uma LLM consultando uma base de conhecimento externa antes de gerar a resposta.
* **Embeddings:** Vetores numéricos que representam o significado semântico de palavras ou textos, permitindo que computadores calculem a "proximidade" de ideias.
* **LangChain:** Framework integrador em Python usado para encadear LLMs, memórias, prompts e ferramentas externas.

---

### 🛠️ Prompts Reutilizáveis para Revisão Coletiva
* 🎯 `“Atuando como um entrevistador técnico de Data Science, crie um quiz de 5 perguntas de múltipla escolha com gabarito comentado sobre o fluxo de treino e teste do Scikit-learn com base nas fontes.”`
* 🎯 `“Gere um resumo executivo em tópicos explicando a diferença prática entre o foco do Scikit-learn (predição) e do Statsmodels (inferência), utilizando trechos dos documentos.”`
* 🎯 `“Crie um guia passo a passo simplificado para explicar o conceito de busca semântica em arquiteturas RAG para uma pessoa leiga.”`
