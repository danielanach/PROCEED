import plotly
import os, sys
import base64
import sklearn
import numpy as np
import pandas as pd
import streamlit as st

# Checkpoint for XGBoost
xgboost_installed = False
try:
    import xgboost
    from xgboost import XGBClassifier
    xgboost_installed = True
except ModuleNotFoundError:
    pass

# Object for dict
class objdict(dict):
    """
    Objdict class to conveniently store a state
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

# Main components
def main_components():
    """
    Expose external CSS and create & return widgets
    """
    # External CSS
    main_external_css = """
        <style>
            hr {margin: 15px 0px !important; background: #ff3a50}
            .footer {position: absolute; height: 50px; bottom: -150px; width:100%; padding:10px; text-align:center; }
            #MainMenu, .reportview-container .main footer {display: none;}
            .btn-outline-secondary {background: #FFF !important}
            .download_link {color: #f63366 !important; text-decoration: none !important; z-index: 99999 !important;
                            cursor:pointer !important; margin: 15px 0px; border: 1px solid #f63366;
                            text-align:center; padding: 8px !important; width: 200px;}
            .download_link:hover {background: #f63366 !important; color: #FFF !important;}
            h1, h2, h3, h4, h5, h6, a, a:visited {color: #f84f57 !important}
            label, stText, p, .caption {color: #035672}
            .css-17eq0hr {background: #035672 !important;}
            .streamlit-expanderHeader {font-size: 16px !important;}
            .css-17eq0hr label, stText, .caption, .css-j075dz, .css-1t42vg8 {color: #FFF !important}
            .css-17eq0hr a {text-decoration:underline;}
            .tickBarMin, .tickBarMax {color: #f84f57 !important}
            .markdown-text-container p {color: #035672 !important}
            .css-xq1lnh-EmotionIconBase {fill: #ff3a50 !important}
            .css-hi6a2p {max-width: 800px !important}

            /* Tabs */
            .tabs { position: relative; min-height: 200px; clear: both; margin: 40px auto 0px auto; background: #efefef; box-shadow: 0 48px 80px -32px rgba(0,0,0,0.3); }
            .tab {float: left;}
            .tab label { background: #f84f57; cursor: pointer; font-weight: bold; font-size: 18px; padding: 10px; color: #fff; transition: background 0.1s, color 0.1s; margin-left: -1px; position: relative; left: 1px; top: -29px; z-index: 2; }
            .tab label:hover {background: #035672;}
            .tab [type=radio] { display: none; }
            .content { position: absolute; top: -1px; left: 0; background: #fff; right: 0; bottom: 0; padding: 30px 20px; transition: opacity .1s linear; opacity: 0; }
            [type=radio]:checked ~ label { background: #035672; color: #fff;}
            [type=radio]:checked ~ label ~ .content { z-index: 1; opacity: 1; }

            /* Feature Importance Plotly Link Color */
            .js-plotly-plot .plotly svg a {color: #f84f57 !important}
        </style>
    """
    st.markdown(main_external_css, unsafe_allow_html=True)

    # Fundemental elements
    widget_values = objdict()
    record_widgets = objdict()

    # Sidebar widgets

    return widget_values, record_widgets

# Generate sidebar elements
# Create new list and dict for sessions
@st.cache(allow_output_mutation=True)
# Saving session info
# Load data
@st.cache(persist=True, show_spinner=True)
def load_data(file_buffer, delimiter):
    """
    Load data to pandas dataframe
    """

    warnings = []
    df = pd.DataFrame()
    if file_buffer is not None:
        if delimiter == "Excel File":
            df = pd.read_excel(file_buffer)

            #check if all columns are strings valid_columns = []
            error = False
            valid_columns = []
            for idx, _ in enumerate(df.columns):
                if isinstance(_, str):
                    valid_columns.append(_)
                else:
                    warnings.append(f'Removing column {idx} with value {_} as type is {type(_)} and not string.')
                    error = True
            if error:
                warnings.append("Errors detected when importing Excel file. Please check that Excel did not convert protein names to dates.")
                df = df[valid_columns]

        elif delimiter == "csv":
            df = pd.read_csv(file_buffer, sep=',')
        elif delimiter == "xls":
            df = pd.read_csv(file_buffer, sep=';')
    return df, warnings

# Show main text and data upload section
def main_text_and_data_upload(state, APP_TITLE):
    st.title(APP_TITLE)
    
    st.markdown("This is the description")

    
    with st.beta_expander("Upload or select sample dataset (*Required)", expanded=True):
        st.info(""" Upload your excel / csv file here. Maximum size is 200 Mb. """)
        st.markdown("""**Note:** Please upload an Excel file or csv file""")
        file_buffer = st.file_uploader("Upload your dataset below", type=["csv", "xlsx"])
        st.markdown("""**Note:** By uploading a file, you agree to our
                    [Apache License](https://github.com/OmicEra/OmicLearn/blob/master/LICENSE).
                    Data that is uploaded via the file uploader will not be saved by us;
                    it is only stored temporarily in RAM to perform the calculations.""")
        
        delimiter = st.selectbox("Determine the delimiter in your dataset", ["Excel File", "csv"])
        df, warnings = load_data(file_buffer, delimiter)
        result = False


        state['df'] = df
        
        

        # Sample dataset / uploaded file selection
        dataframe_length = len(state.df)
        max_df_length = 30

        if 0 < dataframe_length < max_df_length:
            st.markdown("Using the following dataset:")
            st.dataframe(state.df)
            result = st.button("Create Download Link")
        elif dataframe_length > max_df_length:
            st.markdown("Using the following dataset:")
            
            st.info(f"The dataframe is too large, displaying the first {max_df_length} rows.")
            st.dataframe(state.df.head(max_df_length))
            result = st.button("Create Download Link")
        else:
            #st.warning("**WARNING:** No dataset uploaded or selected.")
            pass

        #DOWNLOAD csv File
        if result  == True:
            df_download = pd.DataFrame(df)
            csv = df_download.to_csv(index = False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings
            linko= f'<a href="data:file/csv;base64,{b64}" download="SEQResults.csv">Download csv file</a>'
            st.markdown(linko, unsafe_allow_html=True)
  
    with st.beta_expander("Calculate Results",expanded =True):
       result = st.button("Do Calculation")
       
    #Calculation result
       if result == True:
           st.text("hello")
           st.text(df['PCR1CYCLES'].sum())
    
    return state