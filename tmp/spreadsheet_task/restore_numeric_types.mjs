import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const sheet = workbook.worksheets.getItem("Сравнение моделей");
const values = sheet.getRange("A1:O10").values;

for (let r = 1; r < 9; r++) {
  for (let c = 2; c < 14; c++) {
    const value = values[r][c];
    if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) {
      sheet.getCell(r, c).values = [[Number(value)]];
    }
  }
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path);
const finalValues = sheet.getRange("A1:O10").values;
for (let r = 1; r < 10; r++) {
  for (let c = 2; c < 14; c++) {
    if (typeof finalValues[r][c] === "string" && finalValues[r][c] !== "") {
      throw new Error(`Числовая ячейка осталась текстом: ${r + 1}:${c + 1}`);
    }
  }
}
const check = await workbook.inspect({
  kind: "table",
  range: "Сравнение моделей!A1:O10",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 15,
  maxChars: 12000,
});
console.log(check.ndjson);
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
