import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const source = workbook.worksheets.getItem("Сравнение моделей");
const expectedOld = structuredClone(source.getRange("A1:O9").values);
const expectedOldFormulas = structuredClone(source.getRange("A1:O9").formulas);
const expectedNew = structuredClone(source.getRange("A11:O11").values);

const clean = workbook.worksheets.add("Сравнение моделей — tmp");
clean.getRange("A1:O9").copyFrom(source.getRange("A1:O9"), "all");
clean.getRange("A10:O10").copyFrom(source.getRange("A11:O11"), "all");
for (let c = 0; c < 15; c++) {
  const col = String.fromCharCode(65 + c);
  clean.getRange(`${col}1:${col}10`).format.columnWidth =
    source.getRange(`${col}1:${col}11`).format.columnWidth;
}
for (let r = 0; r < 9; r++) {
  clean.getRange(`A${r + 1}:O${r + 1}`).format.rowHeight =
    source.getRange(`A${r + 1}:O${r + 1}`).format.rowHeight;
}
clean.getRange("A10:O10").format.rowHeight = source.getRange("A11:O11").format.rowHeight;
clean.showGridLines = source.showGridLines;
const newTable = clean.tables.add("A1:O10", true, "ModelMetricsComparisonClean");
newTable.showBandedRows = true;
newTable.showFilterButton = true;

const copiedOld = clean.getRange("A1:O9").values;
const copiedFormulas = clean.getRange("A1:O9").formulas;
for (let r = 0; r < expectedOld.length; r++) {
  for (let c = 0; c < expectedOld[r].length; c++) {
    const same = JSON.stringify(expectedOld[r][c]) === JSON.stringify(copiedOld[r][c]);
    const formulaCell = typeof expectedOldFormulas[r][c] === "string" && expectedOldFormulas[r][c].startsWith("=");
    const roundoff = formulaCell && typeof expectedOld[r][c] === "number" &&
      typeof copiedOld[r][c] === "number" && Math.abs(expectedOld[r][c] - copiedOld[r][c]) < 1e-12;
    const numericEquivalent = Number.isFinite(Number(expectedOld[r][c])) && Number.isFinite(Number(copiedOld[r][c])) &&
      Math.abs(Number(expectedOld[r][c]) - Number(copiedOld[r][c])) < 1e-12;
    if (!same && !roundoff && !numericEquivalent) throw new Error(`Изменилось значение ${r + 1}:${c + 1}: ${expectedOld[r][c]} -> ${copiedOld[r][c]}`);
  }
}
if (JSON.stringify(copiedFormulas) !== JSON.stringify(expectedOldFormulas)) {
  throw new Error("Существующие формулы не сохранились при удалении пустой строки");
}
if (JSON.stringify(clean.getRange("A10:O10").values) !== JSON.stringify(expectedNew)) {
  throw new Error("Новая строка не сохранилась");
}

source.delete();
clean.name = "Сравнение моделей";

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path);
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
const preview = await workbook.render({ sheetName: "Сравнение моделей", autoCrop: "all", scale: 1.5, format: "png" });
await fs.writeFile(
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/tmp/spreadsheet_task/after.png",
  new Uint8Array(await preview.arrayBuffer()),
);
