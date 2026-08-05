import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
const path = "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const sheet = workbook.worksheets.getItem("Сравнение моделей");
const table = sheet.tables.items[0];
console.log(JSON.stringify({ name: table.name, style: table.style, headers: table.showHeaders, banded: table.showBandedRows, used: sheet.getUsedRange().address }));
table.delete();
console.log((await workbook.inspect({kind:"table,region", sheetId:sheet.name, range:"A1:O11", maxChars:5000, tableMaxRows:12, tableMaxCols:15})).ndjson);
