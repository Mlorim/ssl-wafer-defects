import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
const path = "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const sheet = workbook.worksheets.getItem("Сравнение моделей");
console.log(JSON.stringify(sheet.getRange("A1:O10").formulas));
