import sys

import pandas as pd

sys.path.append('../')
import pandas.io.sql as psql
import psycopg2
import re
from datetime import datetime
from tqdm import tqdm

CONST_USER = "postgres"
CONST_SCHEMA_NAME = "public"
today = datetime.now()
lvi_postfix = today.strftime("%Y%m%d")
CONST_KR_TABLE = "test_lvi_prm"


def connect_DB(table, kr_table):

    try:
        connection = psycopg2.connect(user="user",
                                      password="password",
                                      host="host",
                                      port="port",
                                      database="database")

        cursor = connection.cursor()

        cursor.execute(f"SELECT tablename FROM PG_TABLES where tablename like '{table}' order by tablename")
        table_lsts = cursor.fetchall()
        table_name = re.sub('[^a-zA-Z0-9_]', '', table_lsts[-1][0]).strip()
        # print(table_name)
        # exit()

        # cursor.execute(f"SELECT tablename FROM PG_TABLES where tablename like '{kr_table}' order by tablename")
        # kr = cursor.fetchall()

        # print(CONST_KR_TABLE)
        # exit()

        # if kr[0][0] != CONST_KR_TABLE:
        #     cursor.execute(f"ALTER TABLE {kr[0][0]} RENAME TO {CONST_KR_TABLE}")

        #cursor.execute(f"ALTER TABLE {kr_table} RENAME TO {CONST_KR_TABLE}")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    return connection, cursor, table_name

def connect_DB2(table):

    try:
        connection = psycopg2.connect(user="user",
                                      password="password",
                                      host="host",
                                      port="port",
                                      database="database")

        cursor = connection.cursor()

        cursor.execute(f"SELECT tablename FROM PG_TABLES where tablename like '{table}' order by tablename")
        table_lsts = cursor.fetchall()

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    return connection, cursor, table_lsts

def connect_DB3(tablename):

    try:
        connection = psycopg2.connect(user="user",
                                      password="password",
                                      host="host",
                                      port="port",
                                      database="database")

        cursor = connection.cursor()
        df = pd.read_sql(f"SELECT * FROM {tablename}", connection)

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    return connection, cursor, df

def connect_DB4(tablename):

    try:
        connection = psycopg2.connect(user="user",
                                      password="password",
                                      host="host",
                                      port="port",
                                      database="database")

        cursor = connection.cursor()
        df = pd.read_sql(f"SELECT * FROM {tablename}", connection)

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    return connection, cursor, df

def select_dest_code_from_postgresQL(mmsi, destination):
    whatDate = datetime.now()
    yearMonthDay = whatDate.strftime('%Y%m%d')
    table = 'lvi_prm_' + yearMonthDay + '%'
    connection, cursor, table_lsts = connect_DB2(table)
    table_lsts.reverse()
    # print(table_lsts)
    # exit()
    for table_na in table_lsts:
        # print(table_na[0])
        # exit()
        cursor.execute(f"SELECT destination FROM {table_na[0]} WHERE mmsi={mmsi}", connection)
        before_dest = cursor.fetchall()
        print(before_dest)
        if before_dest[0][0] != destination:
            break
    return connection, cursor, before_dest[0][0]

def select_ship_info_from_postgresQL(cur_table, kr_table, col):

    connection, cursor, table_na = connect_DB(cur_table, kr_table)

    tpls1 = psql.read_sql("SELECT imo FROM public.kr_vessel2", connection)
    tpls1 = tuple(tpls1['imo'])
    cur_records = psql.read_sql(f"SELECT * FROM {table_na} WHERE {col} in {tpls1}", connection)
    past_records = psql.read_sql("SELECT * FROM {}".format(CONST_KR_TABLE), connection)

    #records = psql.read_sql(f"SELECT * FROM {table_na}", connection)
    # print("Record selected successfully")
    return connection, cursor, cur_records, past_records

def disconnection(connection, cursor):
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")

# Check table if it is existed
def is_table_existed(ps_connection):
    print(f"Checking table {CONST_KR_TABLE}...")

    isExisted = False
    commands = (
        f"""
            SELECT EXISTS(
                SELECT FROM pg_tables
                WHERE schemaname = '{CONST_SCHEMA_NAME}' AND tablename = '{CONST_KR_TABLE}'
            );
        """)
    try:
        # create a cursor
        cur = ps_connection.cursor()
        # print(commands)
        cur.execute(commands)
        isExisted = cur.fetchone()[0]
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()

    print("Is table {} existed? {}".format(CONST_KR_TABLE, isExisted))

    return isExisted

# Create a new table in the database
def create_table(ps_connection, commands):
    try:
        # create a cursor
        cur = ps_connection.cursor()
        cur.execute(commands)

        # commit the changes
        ps_connection.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()

# Create a new table in the database
def create_kr_ship_table(ps_connection):

    def make_table():
        full_table_name = CONST_SCHEMA_NAME + "." + CONST_KR_TABLE
        print("Creating table ...")
        """ create tables in the PostgreSQL database"""
        commands = (
            f"""
                    CREATE TABLE {full_table_name}
                    (
                        gid serial NOT NULL PRIMARY KEY,
                        mmsi numeric,
                        imo integer,
                        vessel_name character varying(254) COLLATE pg_catalog."default",
                        callsign character varying(254) COLLATE pg_catalog."default",
                        vessel_type character varying(254) COLLATE pg_catalog."default",
                        vessel_type_code integer,
                        vessel_type_cargo character varying(254) COLLATE pg_catalog."default",
                        vessel_class character varying(254) COLLATE pg_catalog."default",
                        length integer,
                        width integer,
                        flag_country character varying(254) COLLATE pg_catalog."default",
                        flag_code integer,
                        destination character varying(254) COLLATE pg_catalog."default",
                        eta character varying(254) COLLATE pg_catalog."default",
                        draught numeric,
                        longitude numeric,
                        latitude numeric,
                        sog numeric,
                        cog numeric,
                        rot numeric,
                        heading numeric,
                        nav_status character varying(254) COLLATE pg_catalog."default",
                        nav_status_code integer,
                        source character varying(254) COLLATE pg_catalog."default",
                        ts_pos_utc character varying(254) COLLATE pg_catalog."default",
                        ts_static_utc character varying(254) COLLATE pg_catalog."default",
                        ts_insert_utc character varying(254) COLLATE pg_catalog."default",
                        dt_pos_utc character varying(254) COLLATE pg_catalog."default",
                        dt_static_utc character varying(254) COLLATE pg_catalog."default",
                        dt_insert_utc character varying(254) COLLATE pg_catalog."default",
                        vessel_type_main character varying(254) COLLATE pg_catalog."default",
                        vessel_type_sub character varying(254) COLLATE pg_catalog."default",
                        message_type integer,
                        eeid numeric,
                        geom geometry(Point,4326),
                        dest_code character varying(254) COLLATE pg_catalog."default",
                        dept_code character varying(254) COLLATE pg_catalog."default",
                        etd character varying(254) COLLATE pg_catalog."default"
                    )

                    TABLESPACE pg_default;

                    ALTER TABLE {full_table_name}
                        OWNER to {CONST_USER};

                    CREATE INDEX index_lvi_mmsi_{lvi_postfix}
                        ON {full_table_name} USING btree
                        (mmsi ASC NULLS LAST)
                        TABLESPACE pg_default;
                    -- Index: index_vessel_name
                    """)
        create_table(ps_connection, commands)

        print("Done of creating table {} ...".format(full_table_name))

    def drop_table(ps_connection, table_name):
        try:
            # create a cursor
            cur = ps_connection.cursor()

            # execute a statement
            cur.execute(f"""DROP TABLE IF EXISTS {table_name}""")
            ps_connection.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if cur:
                cur.close()
    if is_table_existed(ps_connection) == False:
        make_table()
    else:
        drop_table(ps_connection, CONST_KR_TABLE)
        make_table()

# Create kr_table and select data from postgresQL
def extract_data_from_postgresQL(cur_table, kr_table, col):

    connection, cursor, cur_records, past_records = select_ship_info_from_postgresQL(cur_table, kr_table, col)
    # print(cur_records.keys())
    # print(past_records.keys())
    # exit()
    create_kr_ship_table(connection)

    disconnection(connection, cursor)
    
    return cur_records, past_records


def update_table_in_postgresQL(krTableName, tablename):
    try:
        connection = psycopg2.connect(user="user",
                                      password="password",
                                      host="host",
                                      port="port",
                                      database="database")

        cursor = connection.cursor()

        # for kr_lst in kr_lists:
            #insert_query = (f"UPDATE {tablename} SET dest_code='{kr_lst[1]}', dept_code='{kr_lst[2]}'  WHERE imo={kr_lst[0]}")
        insert_query = (f"UPDATE {tablename} as A SET dest_code=B.dest_code, dept_code=B.dept_code\
                        FROM (SELECT imo, dest_code, dept_code FROM {krTableName}) as B where \
                        A.imo = B.imo")
        cursor.execute(insert_query, insert_query)
        connection.commit()

        print("Record updated successfully")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


if __name__ == "__main__":

    tablename = 'member'
    connect, cursor, df = connect_DB3(tablename)

    # mmsi = 352688000
    # destination = "KR PUS"
    # connection, cursor, before_dest = select_dest_code_from_postgresQL(mmsi, destination)
    #
    # print(before_dest)
    # kr_df = pd.read_csv("/Users/jeongtaegun/Desktop/surver_project/trackAndODPairs/pyAPI/in/kr_ship_1877.csv", sep=',', encoding='utf8')
    # ship_imo = tuple(kr_df['IMO No,'])
    # col = 'imo'
    # whatDate = datetime.now()
    # yearMonthDay = whatDate.strftime('%Y%m%d')
    # cur_table = 'lvi_prm_' + yearMonthDay + '%'
    # #print(table)
    # #exit()
    # connection, cursor, cur_records, past_records, kr_tablename = extract_data_from_postgresQL\
    #     (ship_imo, cur_table,col)
    # disconnection(connection, cursor)
    # print(past_records)

