import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const source = "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 12000,
  tableMaxRows: 30,
  tableMaxCols: 20,
  tableMaxCellChars: 180,
});
console.log(summary.ndjson);
const sheet = workbook.worksheets.getItemAt(0);
const used = sheet.getUsedRange();
console.log(JSON.stringify({ sheet: sheet.name, used: used.address, values: used.values }));
const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1.5, format: "png" });
await fs.writeFile(
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/tmp/spreadsheet_task/before.png",
  new Uint8Array(await preview.arrayBuffer()),
);
