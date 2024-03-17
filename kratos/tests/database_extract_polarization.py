#! /usr/bin/env python
#
# Example script to read out parameters from the CR Database
# Change the variable table_entries_to_read_out, and the WHERE clause,
# for your particular query and selection criteria.
# Optionally output the list of event IDs for this query to a text file
#
# Author: Arthur Corstanje, a.corstanje@astro.ru.nl, Nov 2016.
# Hacked for use with python3, May 2022.


from optparse import OptionParser
import pickle
import numpy as np
import re
import base64

try:
    import psycopg2

    have_psycopg2 = True
except ImportError:
    print('No database connection, could not import psycopg2!')
    raise


def decode_sql_output(sql_output):
    output_list = []
    for row in sql_output:
        newRow = []
        for item in row:
            if type(item) == str and item.startswith("base64_"):
                # import pdb; pdb.set_trace()
                decoded_string = base64.b64decode(item[len("base64_"):].encode('latin1'))
                # newRow.append(pickle.loads(item[len("base64_"):].decode("base64")))
                newRow.append(pickle.loads(decoded_string, encoding='latin1'))  # , encoding='bytes'))
            else:
                newRow.append(item)
        output_list.append(newRow)
    return output_list


def print_table(table_header, list_of_rows):
    # get maximum length for table
    array_of_rows = np.array(list_of_rows)
    row_format = ""
    for i in range(len(table_header)):
        item_max_length = len(max(array_of_rows[:, i], key=len))
        item_max_length = max(item_max_length, len(table_header[i]))
        row_format += "{:%d} | " % item_max_length

    table_head = row_format.format(*table_entriesinstituut_to_read_out)

    print(table_head)
    print('-' * len(table_head))
    for row in list_of_rows:
        print(row_format.format(*row))


parser = OptionParser()
parser.add_option("-e", "--eventid", default="81409140", help="Event ID")
# parser.add_option("-s", "--station", default=None, help="Station ID; all stations if not specified")
# parser.add_option("-p", "--polarization", default="0", help="0 (default, = LOFAR w/o applying antenna model) or xyz (ie after applying antenna model)")
parser.add_option("-o", "--outputeventlist", default="", help="Optional output of event list to text file")

(options, args) = parser.parse_args()
eventid = str(options.eventid)
# polarization = str(options.polarization)
# station = options.station

# Connect to database
# Open PostgreSQL database
conn = psycopg2.connect(host='astropg.science.ru.nl', user='crdb', password='crdb', dbname='crdb')
# Get cursor on database
cur = conn.cursor()

# Example 1: Check station error messages for CS004, for events with station CS004 in ERROR and optionally some more conditions
table_entries_to_read_out = ['e.eventID', 'e.status', 's.stationname', 's.status', 's.statusmessage']
sql_start = "SELECT " + ', '.join(table_entries_to_read_out)
# Gives: SELECT e.eventID, e.status, s.stationname, s.status, s.statusmessage
sql = sql_start + """ FROM events AS e
INNER JOIN event_datafile   AS ed ON (e.eventID=ed.eventID)
INNER JOIN datafile_station AS ds ON (ed.datafileID=ds.datafileID)
INNER JOIN stations         AS s ON (ds.stationID=s.stationID)
WHERE (s.stationname='CS004' AND s.status='ERROR' AND e.status='CR_FOUND') ORDER BY e.eventID;"""
# AND e.status='CR_FOUND' AND e.antennaset='LBA_OUTER' AND s.statusmessage LIKE '%[65536]%'


# Example 2: read out crp_median_pulse_snr for CS002 for CR_FOUND, LBA_OUTER events with CS002 status GOOD
want_example_2 = False
if want_example_2:
    table_entries_to_read_out = ['e.eventID', 'lora_energy', 'simulation_energy']
    sql_start = "SELECT " + ', '.join(table_entries_to_read_out)
    # Gives: SELECT e.eventID, s.stationname, sp.crp_median_pulse_snr
    sql = sql_start + """ FROM events AS e
        LEFT JOIN eventparameters AS ep ON (e.eventID=ep.eventID)
        WHERE (e.status='CR_FOUND') ORDER BY e.eventID;"""
    # INNER JOIN event_datafile   AS ed ON (e.eventID=ed.eventID)
    # INNER JOIN datafile_station AS ds ON (ed.datafileID=ds.datafileID)
    # INNER JOIN stations         AS s ON (ds.stationID=s.stationID)
    # INNER JOIN stationparameters AS sp ON (s.stationID=sp.stationID)
    # WHERE (e.status='CR_FOUND') ORDER BY e.eventID;"""

want_example_3 = False
if want_example_3:
    table_entries_to_read_out = ['e.eventID', 'ep.polarization_stokes_i', 'ep.polarization_stokes_i_uncertainty',
                                 'ep.polarization_stokes_q', 'ep.polarization_stokes_q_uncertainty']
    sql_start = "SELECT " + ', '.join(table_entries_to_read_out)
    # Gives: SELECT e.eventID, s.stationname, sp.crp_median_pulse_snr

    # station_clause = 'AND s.stationname={0}'.format(station) if station is not None else ''

    sql = sql_start + """ FROM events AS e
        LEFT JOIN eventparameters AS ep ON (e.eventID=ep.eventID)
        WHERE (e.eventID='{0}');""".format(
        eventid)  # {1} AND p.direction='{2}');""".format(eventid, station_clause, polarization)

    # WHERE (e.status='CR_FOUND' AND e.antennaset='LBA_INNER') ORDER BY e.eventID;"""

    # sql = sql_start + """ FROM events AS e
    #    LEFT JOIN eventparameters AS ep ON (e.eventID=ep.eventID)
    #        INNER JOIN event_datafile   AS ed ON (e.eventID=ed.eventID)
    #        INNER JOIN datafile_station AS ds ON (ed.datafileID=ds.datafileID)
    #        INNER JOIN stations         AS s ON (ds.stationID=s.stationID)
    #        INNER JOIN stationparameters AS sp ON (s.stationID=sp.stationID)
    #        WHERE (e.eventID='81409140') ORDER BY e.eventID;"""
    # WHERE (e.status='CR_FOUND') ORDER BY e.eventID;"""

want_example_4 = True
# Get LBA_INNER events with CR_FOUND and simulation_status='good simulation set'
if want_example_4:
    table_entries_to_read_out = ['e.eventID',
                                 'e.simulation_status',
                                 's.stationname',
                                 's.status',
                                 'sp.crp_dirty_channels'
                                 ]
    sql_start = "SELECT " + ', '.join(table_entries_to_read_out)
    # Gives: SELECT e.eventID, s.stationname, sp.crp_median_pulse_snr
    sql = sql_start + """ FROM events AS e
        LEFT JOIN eventparameters AS ep ON (e.eventID=ep.eventID)
        INNER JOIN event_datafile   AS ed ON (e.eventID=ed.eventID)
        INNER JOIN datafile_station AS ds ON (ed.datafileID=ds.datafileID)
        INNER JOIN stations         AS s ON (ds.stationID=s.stationID)
        INNER JOIN stationparameters AS sp ON (s.stationID=sp.stationID)
        WHERE (e.status='CR_FOUND' AND e.eventid='%s') ORDER BY e.eventID;""" % eventid
    # WHERE (e.status='CR_FOUND') ORDER BY e.eventID;"""
"""
INNER JOIN event_datafile   AS ed ON (e.eventID=ed.eventID)
INNER JOIN datafile_station AS ds ON (ed.datafileID=ds.datafileID)
INNER JOIN stations         AS s ON (ds.stationID=s.stationID)
INNER JOIN stationparameters AS sp ON (s.stationID=sp.stationID)
"""
cur.execute(sql)

# Get SQL output
output_list = cur.fetchall()

# Decode SQL output (it may contain base64-encoded strings)
decoded_output_list = decode_sql_output(output_list)

if len(decoded_output_list) == 0:
    print('No output from query! Exiting.')
    raise SystemExit

print(decoded_output_list)

decoded_output_list = np.array(decoded_output_list)

# Here you can do what you want with decoded_output_list


"""

print_table(table_entries_to_read_out, decoded_output_list)
print('---')
print('There are %d events in this list' % len(output_list))

# Insert your code here to do stuff with the output, assemble statistics etc.

if options.outputeventlist != "":
    # Write event list to output file. Assumes eventID is the first item of each row
    print('Writing event list to %s' % options.outputeventlist)
    outfile = open(options.outputeventlist, 'w')
    for row in decoded_output_list:
        outfile.write("%d\n" % row[0])
    outfile.close()
"""
