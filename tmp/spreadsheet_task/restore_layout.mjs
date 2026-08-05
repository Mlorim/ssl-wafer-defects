import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const sheet = workbook.worksheets.getItem("Сравнение моделей");
const table = sheet.tables.items[0];
table.style = "";
table.showBandedRows = false;

sheet.getRange("A1:O10").format.font = { name: "Calibri", size: 7, color: "#1F1F1F" };
sheet.getRange("A1:O10").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  insideVertical: { style: "thin", color: "#FFFFFF" },
};
sheet.getRange("A1:O1").format.fill = "#4472C4";
sheet.getRange("A1:O1").format.font = { name: "Calibri", size: 7, bold: true, color: "#FFFFFF" };
sheet.getRange("A1:O1").format.horizontalAlignment = "center";
sheet.getRange("A1:O1").format.verticalAlignment = "center";
sheet.getRange("A1:O1").format.wrapText = true;

for (let r = 2; r <= 10; r++) {
  sheet.getRange(`A${r}:O${r}`).format.fill = r % 2 === 0 ? "#C0E3F2" : "#F2F4F8";
  sheet.getRange(`A${r}:O${r}`).format.verticalAlignment = "top";
}
sheet.getRange("A2:B10").format.horizontalAlignment = "left";
sheet.getRange("O2:O10").format.horizontalAlignment = "left";
sheet.getRange("B2:B10").format.wrapText = true;
sheet.getRange("O2:O10").format.wrapText = true;
sheet.getRange("C2:N10").format.horizontalAlignment = "right";
sheet.getRange("C2:N10").format.numberFormat = "0.00%";

sheet.getRange("A1:A10").format.columnWidthPx = 120;
sheet.getRange("B1:B10").format.columnWidthPx = 200;
for (const col of "CDEFGHIJKLMN") sheet.getRange(`${col}1:${col}10`).format.columnWidthPx = 67;
sheet.getRange("O1:O10").format.columnWidthPx = 240;
sheet.getRange("A1:O1").format.rowHeightPx = 30;
for (const r of [2, 3, 4]) sheet.getRange(`A${r}:O${r}`).format.rowHeightPx = 30;
for (const r of [5, 6, 7, 8]) sheet.getRange(`A${r}:O${r}`).format.rowHeightPx = 60;
sheet.getRange("A9:O10").format.rowHeightPx = 65;

const deltas = sheet.getRange("K2:N10");
deltas.conditionalFormats.deleteAll();
deltas.conditionalFormats.add("colorScale", {
  colors: ["#F8696B", "#FFEB84", "#63BE7B"],
  thresholds: ["min", { type: "percentile", value: 50 }, "max"],
});

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1.5, format: "png" });
await fs.writeFile(
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/tmp/spreadsheet_task/after.png",
  new Uint8Array(await preview.arrayBuffer()),
);
