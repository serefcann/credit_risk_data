import pandas as pd
import streamlit as st

conn = st.connection("mysql", type="sql")

col1, _ = st.columns([2, 5])
with col1:
    data_id = st.text_input('Enter Customer ID:', placeholder='e.g., 100002')


st.markdown("---")
c1, c2 = st.columns([3,3],gap='large')

if data_id:
    try:
        query = f"SELECT * FROM application_train WHERE SK_ID_CURR = {int(data_id)}"
        df = conn.query(query)

        if not df.empty:
            row_data = df.loc[0]

            with c1:
                st.subheader("Customer Details")
                st.write(f"**Customer ID:** {row_data['SK_ID_CURR']}")
                st.write(f"**Gender:** {'Female' if row_data['CODE_GENDER'] == 'F' else 'Male'}")
                st.write(f"**Income Type:** {row_data['NAME_INCOME_TYPE']}")
                st.write(f"**Education:** {row_data['NAME_EDUCATION_TYPE']}")
            with c2:
                st.subheader("Financial Details")
                st.write(f"**Income Total:** {row_data['AMT_INCOME_TOTAL']}$")
                st.write(f"**Credit Amount:** {row_data['AMT_CREDIT']}$")
                st.write(f"**Goods Amount:** {row_data['AMT_GOODS_PRICE']}$")
        else:
            st.warning("No data found for this ID.")   


    except ValueError:
        st.error("Please enter a valid numeric ID.")
else:
    st.info("Please enter a customer ID to view details.")
