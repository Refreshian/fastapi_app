# https://stackoverflow.com/questions/78387056/langchain-with-llama3-stuck-at-entering-new-agentexecutor-chain
from langchain_community.llms import Ollama
import pandas as pd
import aiohttp
import torch
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
                        

df = pd.read_excel('/home/dev/fastapi/analytics_app/files/df_join.xlsx')
llm = Ollama(model='llama3')
agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True, allow_dangerous_code=True)
# text = 'Какие самые аудиторные авторы (конолка Авторы), аудитория находится в колонке Аудитория'
text = 'Какие ТОП-20 популярных источников?'
result = agent.invoke(text)