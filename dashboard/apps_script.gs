/**
 * Word Golf — backend Apps Script.
 *
 * Setup (vedi README.md della cartella):
 *  1. Crea un nuovo Google Sheet.
 *  2. Estensioni -> Apps Script.
 *  3. (Consigliato) Sostituisci appsscript.json con quello fornito nella
 *     cartella `dashboard/` per limitare l'OAuth scope al SOLO Sheet contenitore
 *     (altrimenti Google chiede l'accesso a tutti i tuoi Sheet).
 *  4. Incolla questo codice, salva.
 *  5. Deploy -> New deployment -> Web app:
 *       - Execute as: Me
 *       - Who has access: Anyone   <-- importante per accettare POST anonimi
 *  6. Copia l'URL della Web app: e' la `DASHBOARD_URL` che dai agli studenti
 *     e al file dashboard.html.
 *
 * Modello dati: un foglio "partite" con colonne:
 *   timestamp | squadra | start | target | mosse | dettaglio
 */

const FOGLIO = "partite";
const HEADERS = ["timestamp", "squadra", "start", "target", "mosse", "dettaglio"];

/** Restituisce il foglio "partite", creandolo se non esiste. */
function _getSheet() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(FOGLIO);
  if (!sh) {
    sh = ss.insertSheet(FOGLIO);
    sh.appendRow(HEADERS);
  } else if (sh.getLastRow() === 0) {
    sh.appendRow(HEADERS);
  }
  return sh;
}

/** Riceve un record di vittoria dal notebook e lo appende al foglio. */
function doPost(e) {
  try {
    const raw = e.postData && e.postData.contents ? e.postData.contents : "{}";
    const body = JSON.parse(raw);
    const ts = body.timestamp ? new Date(body.timestamp) : new Date();
    const squadra = String(body.squadra || "anonima").slice(0, 80);
    const start = String(body.start || "");
    const target = String(body.target || "");
    const mosse = Number(body.mosse || 0);
    const dettaglio = Array.isArray(body.dettaglio)
      ? body.dettaglio.join(" | ").slice(0, 500)
      : "";

    _getSheet().appendRow([ts, squadra, start, target, mosse, dettaglio]);
    return ContentService
      .createTextOutput(JSON.stringify({ok: true}))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ok: false, error: String(err)}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

/** Restituisce tutte le partite come JSON. Usato dalla dashboard HTML. */
function doGet(e) {
  const sh = _getSheet();
  const lastRow = sh.getLastRow();
  let records = [];
  if (lastRow >= 2) {
    const values = sh.getRange(2, 1, lastRow - 1, HEADERS.length).getValues();
    records = values.map(row => ({
      timestamp: row[0] instanceof Date ? row[0].getTime() : row[0],
      squadra: row[1],
      start: row[2],
      target: row[3],
      mosse: row[4],
      dettaglio: row[5],
    }));
  }
  return ContentService
    .createTextOutput(JSON.stringify({ok: true, records: records}))
    .setMimeType(ContentService.MimeType.JSON);
}

/** Helper manuale per pulire la classifica tra una lezione e l'altra. */
function azzeraClassifica() {
  const sh = _getSheet();
  sh.clear();
  sh.appendRow(HEADERS);
}
