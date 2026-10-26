"""Read public XLSX cell values without executing workbook formulas or external links."""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
NS = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
def sheets(path, selected_names=None):
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(node.itertext()) for node in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('x:si', NS)]
        targets = {node.get('Id'): node.get('Target') for node in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        for sheet in workbook.findall('x:sheets/x:sheet', NS):
            if selected_names is not None and sheet.get('name') not in selected_names:
                continue
            target = targets[sheet.get(R + 'id')]
            member = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = []
            for row in ET.fromstring(archive.read(member)).findall('x:sheetData/x:row', NS):
                values = {}
                for cell in row.findall('x:c', NS):
                    address = cell.get('r')
                    column = ''.join(character for character in address if character.isalpha())
                    value = cell.find('x:v', NS)
                    value = value.text if value is not None else ''
                    if cell.get('t') == 's':
                        value = strings[int(value)]
                    elif cell.get('t') == 'inlineStr':
                        value = ''.join(cell.find('x:is', NS).itertext())
                    if value:
                        values[column] = value
                if values:
                    rows.append({'row': int(row.get('r')), 'cells': values})
            yield sheet.get('name'), rows
