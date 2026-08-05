import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "/Users/mironpomazkov/Desktop/ssl-wafer-defects/outputs/model_metrics_comparison.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const sheet = workbook.worksheets.getItem("Сравнение моделей");
const beforeValues = structuredClone(sheet.getRange("A1:O9").values);
const beforeFormulas = structuredClone(sheet.getRange("A1:O9").formulas);
const table = sheet.tables.items[0];

const paperAccuracy = 0.984;
const paperF1 = 0.984;
const myAccuracy = 0.9690911482309614;
const myPrecision = 0.9672287415820354;
const myRecall = 0.9700494371767568;
const myF1 = 0.9682758652804893;

const newRow = [
  "ViT-Tiny/16",
  "WM-38K; 38 классов; leakage-safe stratified 72/8/20; ImageNet pretraining; 224×224",
  paperAccuracy,
  null,
  null,
  paperF1,
  myAccuracy,
  myPrecision,
  myRecall,
  myF1,
  myAccuracy - paperAccuracy,
  null,
  null,
  myF1 - paperF1,
  "В статье не описаны отдельная validation-выборка и полный набор гиперпараметров. Наш checkpoint выбран без доступа к test; оставшийся разрыв может быть связан с другим разбиением и подбором настроек.",
];
const usedAddress = sheet.getUsedRange().address;
if (usedAddress.endsWith("O9")) {
  table.rows.add(null, [[]]);
} else if (usedAddress.endsWith("O11")) {
  table.resize(sheet.getRange("A1:O10"));
  sheet.getRange("A11:O11").clear({ applyTo: "all" });
} else if (!usedAddress.endsWith("O10")) {
  throw new Error(`Неожиданный диапазон листа: ${usedAddress}`);
}
sheet.getRange("A10:O10").values = [newRow];

const afterValues = sheet.getRange("A1:O9").values;
const afterFormulas = sheet.getRange("A1:O9").formulas;
if (JSON.stringify(afterValues) !== JSON.stringify(beforeValues)) {
  const diffs = [];
  for (let r = 0; r < beforeValues.length; r++) {
    for (let c = 0; c < beforeValues[r].length; c++) {
      const formulaCell = typeof beforeFormulas[r][c] === "string" && beforeFormulas[r][c].startsWith("=");
      const numericRoundoff = formulaCell && typeof beforeValues[r][c] === "number" &&
        typeof afterValues[r][c] === "number" && Math.abs(beforeValues[r][c] - afterValues[r][c]) < 1e-12;
      if (JSON.stringify(beforeValues[r][c]) !== JSON.stringify(afterValues[r][c]) && !numericRoundoff) {
        diffs.push({ r: r + 1, c: c + 1, before: beforeValues[r][c], after: afterValues[r][c] });
      }
    }
  }
  if (diffs.length) {
    throw new Error(`Изменились существующие значения A1:O9: ${JSON.stringify(diffs)}`);
  }
}
if (JSON.stringify(afterFormulas) !== JSON.stringify(beforeFormulas)) {
  throw new Error("Изменились существующие формулы A1:O9");
}

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
const preview = await workbook.render({
  sheetName: sheet.name,
  autoCrop: "all",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  "/Users/mironpomazkov/Desktop/ssl-wafer-defects/tmp/spreadsheet_task/after.png",
  new Uint8Array(await preview.arrayBuffer()),
);
