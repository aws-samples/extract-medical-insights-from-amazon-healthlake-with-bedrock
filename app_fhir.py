import json
import streamlit as st
import time
import boto3
import pandas as pd
from anthropic import Anthropic
CLAUDE = Anthropic()
import multiprocessing
import subprocess
import shutil
import os
import codecs
import uuid
import io
from botocore.config import Config
from botocore.exceptions import ClientError
import random
ATHENA=boto3.client('athena')
GLUE=boto3.client('glue')
S3=boto3.client('s3')
from botocore.config import Config
config = Config(
    read_timeout=120,
    retries = dict(
        max_attempts = 10
    )
)
BEDROCK=boto3.client(service_name='bedrock-runtime',region_name='us-east-1',config=config)
st.set_page_config(page_icon=None, layout="wide")
with open('pricing.json','r',encoding='utf-8') as f:
    pricing_file = json.load(f)
with open('config.json', 'r',encoding='utf-8') as f:
    config_file = json.load(f)

ATHENA_WORKGROUP_BUCKET_NAME = config_file["athena-workgroup-bucket-name"]

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'token' not in st.session_state:
    st.session_state['token'] = 0
if 'button' not in st.session_state:
    st.session_state['button'] = False
if 'summary' not in st.session_state:
    st.session_state['summary'] = ""
if 'message' not in st.session_state:
    st.session_state['message'] = []
if 'final_summary' not in st.session_state:
    st.session_state['final_summary'] = ""
if 'summary_1' not in st.session_state:
    st.session_state['summary_1'] = ""
if 'final_summary_1' not in st.session_state:
    st.session_state['final_summary_1'] = ""
if 'fhir_summary' not in st.session_state:
    st.session_state['fhir_summary'] = None
if 'fhir_tables' not in st.session_state:
    st.session_state['fhir_tables'] = None
if 'cost' not in st.session_state:
    st.session_state['cost'] = 0
import json

@st.cache_resource
def token_counter(path):
    tokenizer = LlamaTokenizer.from_pretrained(path)
    return tokenizer

@st.cache_data
def get_database_list(catalog):
    response = ATHENA.list_databases(
        CatalogName=catalog,

    )
    db=[x['Name'] for x in response['DatabaseList']]
    return db

def read_s3_file_to_df(bucket_name,key, max_retries=5, initial_delay=1):
    """
    Reads a file from Amazon S3 into a pandas DataFrame with retry logic.

    Args:
        bucket_name (str): Name of bucket file exist in.
        key (str): Name of file path in S3.
        max_retries (int, optional): The maximum number of retries in case of eventual consistency issues. Default is 5.
        initial_delay (int, optional): The initial delay in seconds before retrying. Default is 1.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the S3 file.

    Raises:
        FileNotFoundError: If the file is not found after the maximum number of retries.
    """
    # Create an S3 client
    s3 = boto3.client('s3')
    delay = initial_delay

    for retry in range(max_retries):
        try:
            # Download the file to memory
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            file_bytes = obj['Body'].read()

            # Read the bytes into a pandas DataFrame
            df = pd.read_csv(io.BytesIO(file_bytes))
            return df

        except s3.exceptions.NoSuchKey:
            if retry == max_retries - 1:
                raise FileNotFoundError(f"File {s3_uri} not found after maximum retries.")
            else:
                delay *= 1.5  # Exponential backoff
                print(f"File not found, retrying in {delay} seconds...")
                time.sleep(delay)

@st.cache_data
def get_table_context(sql, sql2, _prompt,_params):
    response=athena_query_func(sql, _params)
    results=athena_querys(response,sql, _prompt, _params)    
    response1=athena_query_func(sql2, _params)
    results=athena_querys(response,sql2, _prompt, _params)
    query_result=read_s3_file_to_df(ATHENA_WORKGROUP_BUCKET_NAME, f"{response['QueryExecutionId']}.csv", max_retries=5, initial_delay=1).to_csv()
    query_result2=read_s3_file_to_df(ATHENA_WORKGROUP_BUCKET_NAME, f"{response1['QueryExecutionId']}.csv", max_retries=5, initial_delay=1).to_csv()

    return query_result,query_result2


def bedrock_streemer(params,response, handler):
    text=''
    for chunk in response['stream']:       
        if 'contentBlockStart' in chunk:
            tool = chunk['contentBlockStart']['start']['toolUse']       
        elif 'contentBlockDelta' in chunk:
            delta = chunk['contentBlockDelta']['delta']       
            if 'text' in delta:
                text += delta['text']       
                if handler:
                    handler.markdown(text.replace("$","USD ").replace("%", " percent"))
        # elif 'contentBlockStop' in chunk:              
        #     text = ''

        elif 'messageStop' in chunk:
            stop_reason = chunk['messageStop']['stopReason']
        elif "metadata" in chunk:
            st.session_state['input_token']=chunk['metadata']['usage']["inputTokens"]
            st.session_state['output_token']=chunk['metadata']['usage']["outputTokens"]
            latency=chunk['metadata']['metrics']["latencyMs"]
            pricing=st.session_state['input_token']*pricing_file[f"{params['model']}"]["input"]+st.session_state['output_token'] *pricing_file[f"{params['model']}"]["output"]
            st.session_state['cost']+=pricing             
    return text

def bedrock_claude_(params,chat_history,system_message, prompt,model_id,image_path, handler):
    chat_history_copy = chat_history[:]
    content=[]
    if image_path:       
        if not isinstance(image_path, list):
            image_path=[image_path]      
        for img in image_path:
            s3 = boto3.client('s3',region_name="us-east-1")
            match = re.match("s3://(.+?)/(.+)", img)            
            image_name=os.path.basename(img)
            _,ext=os.path.splitext(image_name)
            if "jpg" in ext: ext=".jpeg"           
            bucket_name = match.group(1)
            key = match.group(2)    
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            bytes_image=obj['Body'].read()            
            content.extend([{"text":image_name},{
              "image": {
                "format": f"{ext.lower().replace('.','')}",
                "source": {"bytes":bytes_image}
              }
            }])

    content.append({       
        "text": prompt
            })
    chat_history_copy.append({"role": "user",
            "content": content})
    system_message=[{"text":system_message}]
    response = BEDROCK.converse_stream(messages=chat_history_copy, modelId=model_id,inferenceConfig={"maxTokens": 2000, "temperature": 0.1,},system=system_message)
    answer=bedrock_streemer(params,response, handler) 
    return answer


def _invoke_bedrock_with_retries(params,conversation_history, system_prompt, question, model_id, image_path=None, handler=None):
    max_retries = 10
    backoff_base = 2
    max_backoff = 3  # Maximum backoff time in seconds
    retries = 0

    while True:
        try:
            response = bedrock_claude_(params,conversation_history, system_prompt, question, model_id, image_path, handler)
            return response
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            elif e.response['Error']['Code'] == 'ModelStreamErrorException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            elif e.response['Error']['Code'] == 'EventStreamError':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            elif e.response['Error']['Code'] == 'ValidationException':
               
                raise e
            else:
                # Some other API error, rethrow
                raise

def query_llm(params,prompts,system_prompt):
    model_id=params['model']
    output=_invoke_bedrock_with_retries(params,[], system_prompt, prompts, model_id, [])
    return output

def summary_llm(prompt,params, system_prompt,handler):
    import json
    model_id=params['summary-model']
    answer =_invoke_bedrock_with_retries(params,[], system_prompt, prompt, model_id, [], handler)
    return answer


def athena_query_func(statement, params):
    response = ATHENA.start_query_execution(
        QueryString=statement,
        # ClientRequestToken='string',
        QueryExecutionContext={
            'Database': params["db"],
            'Catalog': 'AwsDataCatalog'
        },
    )
    return response

def error_control(failed_attempts, statement, error, params):

    system_prompt="You are an expert SQL deugger for Amazon Athena"
    prompts=f'''Here is the schema of a table:
<schema>
{params['schema']}
</schema>

Here are sample rows from that table:
<sample row>
{params['sample']}
</sample row>

Here is the sql statement that threw the error below:
<sql>
{statement}
</sql>

Here is the error to debug:
<error>
{error}
</error>

First understand the error and think about how you can fix the error.
Use the provided schema and sample row to guide your thought process for a solution.
Do all this thinking inside <thinking></thinking> XML tags.This is a space for you to write down relevant content and will not be shown to the user.

Once your are done debugging, provide the the correct SQL statement without any additional text.

When generating the correct SQL statement:

- Avoid using JSON parsing functions like JSON_EXTRACT_SCALAR, etc. as they may not be supported.

- SQL engine is Amazon Athena.

- FHIR SQL queries use 1-based indexing for arrays, not 0-based.

Format your response as:

<sql> Correct SQL Statement </sql> '''
    model_id=params['model']
    output=_invoke_bedrock_with_retries(params,[], system_prompt, prompts, model_id, [])

    return output

def chunk_csv_rows(csv_rows, max_token_per_chunk=10000):
    header = csv_rows[0]  # Assuming the first row is the header
    csv_rows = csv_rows[1:]  # Remove the header from the list
    current_chunk = []
    current_token_count = 0
    chunks = []
    header_token=CLAUDE.count_tokens(header)
    for row in csv_rows:
        token = CLAUDE.count_tokens(row)  # Assuming that the row is a space-separated CSV row.

        if current_token_count + token+header_token <= max_token_per_chunk:
            current_chunk.append(row)
            current_token_count += token
        else:
            if not current_chunk:
                raise ValueError("A single CSV row exceeds the specified max_token_per_chunk.")
            header_and_chunk=[header]+current_chunk
            chunks.append("\n".join([x for x in header_and_chunk]))
            current_chunk = [row]
            current_token_count = token

    if current_chunk:
        last_chunk_and_header=[header]+current_chunk
        chunks.append("\n".join([x for x in last_chunk_and_header]))

    return chunks



athena=boto3.client('athena')
def athena_querys(response,q_s,prompt,params):
    if 'failure' in response:
        return {'failed':response, 'sql':q_s,'error control':1}
    max_execution=10
    execution_id = response['QueryExecutionId']
    state = 'RUNNING'
    error_dict={}
    while (max_execution > 0 and state in ['RUNNING', 'QUEUED']):
        max_execution = max_execution - 1
        response = ATHENA.get_query_execution(QueryExecutionId = execution_id)
        if 'QueryExecution' in response and \
                'Status' in response['QueryExecution'] and \
                'State' in response['QueryExecution']['Status']:
            state = response['QueryExecution']['Status']['State']
            if state == 'FAILED':                
                # print("\nBAD SQL:\n")
                # print(q_s)
                error=response['QueryExecution']['Status']['AthenaError']['ErrorMessage']
                # print("\nERROR:")
                # print(error)
                error_dict[max_execution+10]={'failed_query':q_s,'error':error}
                answer=error_control(error_dict,q_s, error, params)
                # print("\nDEBUGGIN...")
                # print(answer)
                idx1 = answer.index('<sql>')
                idx2 = answer.index('</sql>')
                answer=answer[idx1 + len('<sql>') + 1: idx2]                 
                # response=athena_query_func(answer, params)
                response, answer=athena_query_with_self_correction(prompt,answer, params,error_dict)
                return {'result':response, 'sql':answer,'error control':1}
            elif state == 'SUCCEEDED':
                return state


@st.cache_data
def get_tables(database):
    tab=GLUE.get_tables(DatabaseName=database)
    tables=[x["Name"] for x in tab['TableList']]
    return tables

def athena_query_with_self_correction(question,q_s, params, error_bank=None, max_retries=5):
    """
    Execute an Athena query genereated by an LLM with retry logic and error handling.

    Args:
        question (str): user question.
        q_s (str): generated sql syntax.       
        params (list): A list of parameters to pass to the athena_query_func.
        max_retries (int, optional): The maximum number of retries in case of an exception. Default is 5.

    Returns:
        The response and generated sql tuple.
    """
    count = 0    
    if not error_bank:
        error_bank={}
    while count < max_retries:
        try:
            response = athena_query_func(q_s, params)
            return response, q_s
        except Exception as e:
            print(count)
            error_bank[count+1]={'failed_query':q_s,'error':e}
            q_s = error_control(error_bank, q_s, e, params)
            idx1 = q_s.index('<sql>')
            idx2 = q_s.index('</sql>')
            q_s = q_s[idx1 + len('<sql>') + 1: idx2]
            count += 1
            if count==max_retries:
                try:
                    response = athena_query_func(q_s, params)
                except:            
                    response={"failure":f"Error Message: Could not successfully generate a SQL query to get patients medical data in {max_retries} attempts. Please let teh user know so they can try manually."}
                return response, q_s


def db_summary(params):
    sql=f"SELECT * FROM information_schema.columns WHERE table_name='{params[0]}' AND table_schema='{params[1]['db']}'"
    sql2=f"SELECT * from {params[1]['db']}.{params[0]} LIMIT 3"
    patient_id=params[1]['id']
    question=f"Query all information on patient {patient_id}?"
    schema, schema_example =get_table_context(sql, sql2,question, params[1])
    params[1]['schema']=schema
    params[1]['sample']=schema_example
    
    system_prompt="You are an expert SQl developer that generates syntaically correct SQL queries to be executed on Amazon Athena."
    prompts=f'''Here is the schema of the Amazon Healthlake table in CSV format:
<table_schema>
{schema}
</table_schema>

Here are sample row(s) from the table:
<sample_rows>
{schema_example}
</sample_rows>

Question: {question}

When providing a response to the question:
1. Avoid using JSON parsing functions like JSON_EXTRACT_SCALAR, etc. as they may not be supported.
2. FHIR SQL queries use 1-based indexing for arrays, not 0-based.

In your response, provide a single syntactically correct Amazon Athena SQL statement to answer the question. Format your SQL statement as:
<sql>
SQL statement 
</sql>
If there is no patient identifier or patient-related information in the given table schema, format your response as:
<sql>
n/a
</sql>'''
    input_token=CLAUDE.count_tokens(prompts)
    print("bedrock")
    q_s=query_llm(params[1],prompts,system_prompt)
    idx1 = q_s.index('<sql>')
    idx2 = q_s.index('</sql>')
    q_s=q_s[idx1 + len('<sql>') + 1: idx2]
    if q_s.strip() in ['n/a', 'N/A']:
        message=f"Notification Message: The current {params[0]} table does not appear to have any direct link or information on {patient_id}. Please inform the user of this."
    else:
        response, q_s = athena_query_with_self_correction(question, q_s, params[1])

        if 'failure' in response:
            message=response['failure']            
        else:
            results=athena_querys(response,q_s,question, params[1])
            message=None
            retry_count=5
            if isinstance(results, dict):
                i=0
                while isinstance(results, dict):
                    if i<retry_count:  
                        if 'failed' in results:
                            message=f"Error Message: Could not successfully generate a SQL query to get patients medical data in {i} attempts. Please let the user know so they can try manually"
                            break
                        response=results['result']
                        q_s=results['sql']
                        results=athena_querys(response,q_s,question,params[1])
                        i+=1                        
                    if i==retry_count:
                        message=f"Error Message: Could not successfully generate a SQL query to get patients medical data in {i} attempts. Please let the user know so they can try manually"
                        break

    if not message:        
        query_result=read_s3_file_to_df(ATHENA_WORKGROUP_BUCKET_NAME, f"{response['QueryExecutionId']}.csv", max_retries=5, initial_delay=1)        
        fhir_table={params[0]:query_result}
        csv_result=query_result.to_csv()
        print('done')
        
        
    elif message:
        print(message)
        csv_result=message
        fhir_table={params[0]:pd.DataFrame()}
    input_token=CLAUDE.count_tokens(csv_result)
    if input_token>100000:    
        csv_rows=csv_result.split('\n')
        chunk_rows=chunk_csv_rows(csv_rows, max_token_per_chunk=50000)
        initial_summary=[]
        for chunk in chunk_rows:
            system_prompt="You are a medical expert great at analyzing patient data in FHIR format"
            prompts=f'''Here is a patient's {patient_id} medical data:
<medical data>
{chunk}
</medical data>

Provide a detailed technical summary of patient {patient_id} from the above medical data ONLY. DO NOT make any assumptions or add any information not explicitly stated in the medical data.
Do not ask any clarification questions.
If the data is empty, respond with "no relevant information". If the data is nor comma seperated and contains an "Error" or "Notification" message, respond with the message. '''
            initial_summary.append(summary_llm(prompts,params[1], system_prompt,False))
        system_prompt="You are a great summarizer with a medical background."
        prompts = f'''Here is a list of summaries from a given FHIR resource of patient {patient_id} medical record.
<summaries>
{initial_summary}
</summaries>

The summaries are in a list format, collectively containing full information on the patients given FHIR resource.
Review the summaries in great detail.        
Merge the summaries into a single coherent and cohesive narrative of the patients medical file. '''
        summary=summary_llm(prompts,params[1], system_prompt,False)
        summary={params[0]:summary}
    else:
        system_prompt="You are a medical expert great at analyzing patient data in FHIR format"
        prompts=f'''Here is patient's {patient_id} medical data:
<medical data>
{csv_result}
</medical data>

Provide a detailed technical summary of patient {patient_id} from the above medical data ONLY. 
Capture every relevant medical information and associated dates if available.
Do not ask any clarification questions or make up any information.
If the data is empty, respond with "no relevant information". If the data is nor comma seperated and contains an "Error" or "Notification" message, respond with the message. '''
        summary=summary_llm(prompts,params[1], system_prompt,False)
        summary={params[0]:summary}
    return summary,fhir_table

def struct_summary(params):
    prompts=""
    if st.session_state['button']:
        table_names=params['table']
        table_names=[[x]+[params] for x in table_names]
        import multiprocessing    
        # Define the number of concurrent invocations
        num_concurrent_invocations = len(table_names)

        # Create a multiprocessing pool
        pool = multiprocessing.Pool(processes=num_concurrent_invocations)    
        results=pool.map(db_summary,  table_names)
        pool.close()
        pool.join()
        # Unpack the results into separate lists
        summary = [result[0] for result in results]
        fhir_table = [result[1] for result in results]
        
        fhir_keys=[list(x.keys())[0] for x in summary]
        fhir_values=[list(x.values())[0] for x in summary]
        fhir_table_keys=[list(x.keys())[0] for x in fhir_table]
        fhir_table_values=[list(x.values())[0] for x in fhir_table]
        st.session_state['fhir_summary']= dict(zip(fhir_keys, fhir_values))
        st.session_state['fhir_tables']=dict(zip(fhir_table_keys, fhir_table_values))

        if 'prompt 1' in params['template']:
            prompts = f'''Please maintain an active voice tone.
Here are summaries from various FHIR resources of a patient's medical history:
<summaries>
{summary}
</summaries>

Please do the following:
1. Read through the provided summaries from various FHIR resources in detail.
2. Merge the summaries into a single coherent and cohesive paragraph narrative of the patients medical history. 
3. The summary should provide a longitudinal view of the patient's clinical activity based on various ingested CCDAs/FHIR resources.
4. Highlight any abnormal test/lab result findings in your summary if available.
5. Do not infer conditions not explicitly mentioned
6. Exclude dismissed or ruled out conditions. '''
        elif 'prompt 2' in params['template']:
            prompts = f'''Generate a comprehensive summary of patient's medical history from the provided summaries from various FHIR resources of a patient's medical records. 

<summaries>
{summary}
</summaries>

Please do the following:
- Read through the provided summaries from various FHIR resources in detail.
- Identify the patient's major medical conditions, both current and historical.
- Note any procedures, hospitalizations, specialist visits related to these conditions.  
- Mention onset, diagnosis dates, and treatment details for key conditions.
- Ensure accuracy - do not infer conditions not explicitly mentioned.
- Exclude dismissed or ruled out conditions.
- Summarize concisely into a paragraph. '''
       
        st.session_state['summary_1'] = summary        
        
    st.session_state['button']=False
    colm1,colm2=st.columns([1,1])
  
    with colm1:
        container1=st.empty()
        stream_handler = None #StreamHandler(container1)        
        if prompts:
            system_prompt="You are a medical expert."
            final_summary=summary_llm(prompts,params, system_prompt,container1)#, [stream_handler])
            st.session_state['final_summary_1']=final_summary
            st.rerun()
        if st.session_state['final_summary_1']:
            st.header("Patients Consolidated Summary",divider='red')
            st.markdown(st.session_state['final_summary_1'])
            with st.expander(label="**FHIR Section Summary**"):                  
                fhir_keys=list(st.session_state['fhir_summary'].keys())                
                header_holder={}
                tab_objects = st.tabs([f"**{x.upper()}**" for x in fhir_keys])
                # Assign each tab object to the corresponding key in the dictionary
                for i, tab_obj in enumerate(tab_objects):
                    header_holder[fhir_keys[i]] = tab_obj 
                for key in fhir_keys:
                    with header_holder[key]:
                        st.markdown(st.session_state['fhir_summary'][key], unsafe_allow_html=True )
            with st.expander(label="**FHIR Section Tables**"): 
                fhir_table_keys=list(st.session_state['fhir_tables'].keys())
                header_holder={}
                tab_objects = st.tabs([f"**{x.upper()}**" for x in fhir_table_keys])
                # Assign each tab object to the corresponding key in the dictionary
                for i, tab_obj in enumerate(tab_objects):
                    header_holder[fhir_table_keys[i]] = tab_obj 
                for key in fhir_table_keys:
                    with header_holder[key]:
                        st.dataframe(st.session_state['fhir_tables'][key])
    with colm2:
        with st.container(height=500):
            if st.session_state['final_summary_1']:            
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):   
                        if "```" in message["content"]:
                            st.markdown(message["content"],unsafe_allow_html=True )
                        else:
                            st.markdown(message["content"].replace("$", "\$"),unsafe_allow_html=True )


                if prompt := st.chat_input("Whats up?"):        
                    st.session_state.messages.append({"role": "user", "content": prompt})        
                    with st.chat_message("user"):        
                        st.markdown(prompt.replace("$", "\$"),unsafe_allow_html=True )
                    with st.chat_message("assistant"): 
                        message_placeholder = st.empty()      
                        system_prompt="You are an medical expert."
                        prompts=f'''Here is a medical record of a patient:
    <record>
    {st.session_state['summary_1']}
    </record>                   

    Review the patients medical record thoroughly. 
    Provided an answer to the question if available in the medical record.
    Do not include or reference quoted content verbatim in the answer
    If the question cannot be answered by the document, say so.

    Question: {prompt}? '''
                        output_answer=summary_llm(prompts,params, system_prompt,message_placeholder) 
                        message_placeholder.markdown(output_answer.replace("$", "\$"),unsafe_allow_html=True )
                        st.session_state.messages.append({"role": "assistant", "content": output_answer}) 
                        st.rerun()

@st.cache_data
def get_patient_id(sql, _params):    
    response=athena_query_func(sql, _params)
    results=athena_querys(response,sql,'',_params) 
    query_result=read_s3_file_to_df(ATHENA_WORKGROUP_BUCKET_NAME, f"{response['QueryExecutionId']}.csv", max_retries=5, initial_delay=1)
    return list(query_result.iloc[:,0])

def app_sidebar():
    with st.sidebar:
        st.text_input('Bedrock Usage Cost', str(round(st.session_state['cost'],2)))
        st.write('## How to use:')
        description = """A simple Interface for Querying and Chatting with your DataBase
        """
        st.write(description)
        st.write('---')
        st.write('### User Preference')
        models=[
 "anthropic.claude-3-sonnet-20240229-v1:0",
  "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "anthropic.claude-3-haiku-20240307-v1:0",
  'anthropic.claude-instant-v1', 
  'anthropic.claude-v2']
        model=st.selectbox('Model', models, index=1)
        summary_model=st.selectbox('Summary Model', models, index=0)
        db=get_database_list('AwsDataCatalog')
        database=st.selectbox('Select Database',options=db)#,index=6)
        # st.write(database)
        tab=get_tables(database)            
        tables=st.multiselect(
            'FHIR resources',
            tab) 
        sql='''SELECT id FROM healthlake_db.patient;'''
        params={'table':tables,'db':database,'model':model,'summary-model':summary_model}
        patient_ids=get_patient_id(sql, params)
        patient_id=st.selectbox('Patients ID',options=patient_ids)
        prompt_template=st.selectbox('Prompt Templates',options=['prompt 1','prompt 2'])
        st.session_state['button']=st.button('Summarize', type='primary', key='structured')
        params['id']=patient_id
        params['template']=prompt_template
        return params
        
        
        
def main():
    params=app_sidebar()   
    struct_summary(params)
    st.session_state['button']=False

if __name__ == '__main__':
    main()       
