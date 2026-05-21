# Dashboard live — Word Golf

Mini-dashboard che si aggiorna in tempo reale mentre le squadre giocano dai loro Colab.

Setup ~10 minuti **una volta sola**. Le lezioni successive richiedono solo di azzerare la classifica.

## Architettura

```
┌──────────┐       ┌──────────────┐       ┌─────────────┐
│ Colab    │       │ Apps Script  │       │ Google      │
│ studente │──POST→│   Web App    │──────→│ Sheet       │
└──────────┘       └──────────────┘       └─────────────┘
                          ▲                       │
                          │                       │
                          │                       ▼
┌──────────────────┐      │              ┌─────────────────┐
│ dashboard.html   │──GET─┘              │ riga aggiunta   │
│ (browser prof)   │                     │ alla classifica │
└──────────────────┘                     └─────────────────┘
```

Tutto gira su infrastruttura Google gratuita. Zero hosting, zero credenziali in chiaro lato studente.

## Setup passo-passo

### 1. Crea il Google Sheet

1. Vai su [sheets.new](https://sheets.new).
2. Rinominalo come ti pare (es. "Word Golf — classifica").
3. **Non serve creare colonne o fogli**: Apps Script lo farà al primo POST.

### 2. Incolla l'Apps Script

1. Dal Sheet: **Estensioni → Apps Script**.
2. Si apre l'editor. Cancella il `function myFunction() {}` di default.
3. Apri il file [`apps_script.gs`](apps_script.gs) di questa cartella, copia tutto, incolla nell'editor.
4. Salva (icona del floppy o `⌘S` / `Ctrl+S`).

### 2-bis. (Consigliato) Restringi le autorizzazioni al solo Sheet della lezione

Senza questo passaggio, Google al primo deploy ti chiederà l'accesso a **tutti i tuoi
Google Sheet**. Il codice dello script tocca solo il foglio contenitore, ma l'OAuth scope
di default è largo. Per restringerlo:

1. Nell'editor Apps Script, clicca l'icona ⚙️ (Project Settings) in basso a sinistra.
2. Attiva **"Show 'appsscript.json' manifest file in editor"**.
3. Torna su `<>` (Editor): ora vedi anche il file `appsscript.json` nella sidebar.
4. Apri [`appsscript.json`](appsscript.json) di questa cartella, copia tutto,
   incolla sovrascrivendo l'esistente.
5. Salva.

Da questo momento il prompt di autorizzazione dirà solo "vedere, modificare,
creare ed eliminare **questo Spreadsheet**" — non più "tutti i tuoi Spreadsheet".

### 3. Deploy come Web App

1. In alto a destra: **Deploy → New deployment**.
2. Tipo di deploy: clicca la ⚙️ accanto a "Select type" → **Web app**.
3. Compila:
   - **Description**: `wordgolf-v1` (qualsiasi cosa).
   - **Execute as**: `Me` (il tuo account).
   - **Who has access**: **Anyone** ← critico per accettare POST dagli studenti.
4. Clicca **Deploy**.
5. Google chiede l'autorizzazione: accetta. La prima volta vedrai un warning "Google hasn't verified this app" — è normale per script personali. Clicca "Advanced" → "Go to ... (unsafe)" → "Allow".
6. Copia l'**URL della Web app**: è una stringa tipo
   `https://script.google.com/macros/s/AKfycb.../exec`.
   👉 **Questo è quello che dai agli studenti e che incolli nella dashboard.**

### 4. Apri la dashboard

1. Apri [`dashboard.html`](dashboard.html) **direttamente nel browser** (doppio click o `File → Open` da Chrome/Firefox).
2. Incolla l'URL della Web app nel campo "URL Apps Script", premi **Connetti**.
3. Dovresti vedere `0 partite • aggiornato HH:MM:SS` con il pallino verde.
4. Da ora basta proiettare questa pagina.

### 5. Dai l'URL agli studenti

Sulla lavagna o nello slide iniziale, scrivi la URL della Web app.
Ogni coppia la incolla nella loro variabile `DASHBOARD_URL = "..."` nel Modulo 4 del notebook.

### 6. Tra una lezione e l'altra

Per azzerare la classifica:
- **Estensioni → Apps Script** (sul tuo Sheet)
- Dal menu della funzione, scegli `azzeraClassifica`
- Clicca ▶️ (Run)

## Risoluzione problemi

**La dashboard mostra "errore: Failed to fetch"**
- Hai messo "Who has access: Anyone" nel deploy? Se no, ridoplora.
- Apri la URL della Web app direttamente in una nuova tab: dovresti vedere
  `{"ok":true,"records":[]}` (o records popolati). Se vedi una pagina di login,
  il deploy non è "Anyone".

**Gli studenti non vedono il loro nome comparire dopo aver vinto**
- Controlla nel Sheet che la riga sia stata aggiunta.
- Se nel Sheet manca: chiedi allo studente di stampare nel notebook
  `os.environ.get("WORDGOLF_DASHBOARD_URL")` o `llmlego_scuola.golf.DASHBOARD_URL`
  per verificare che la URL sia configurata.
- Se nel Sheet c'è ma la dashboard non aggiorna: ricarica `dashboard.html` (F5).

**Vedo "Google hasn't verified this app" agli studenti**
- Non li sta avvisando l'Apps Script ma il loro Colab in qualche altro contesto.
  L'Apps Script con "Anyone" non richiede consent agli utenti.

## Test locale (senza alunni)

Per verificare che tutto funzioni prima della lezione, da Python:

```python
import urllib.request, json
url = "https://script.google.com/macros/s/..../exec"
data = json.dumps({
    "squadra": "test",
    "start": "cat",
    "target": "eagle",
    "mosse": 3,
    "timestamp": 1234567890000,
    "dettaglio": ["+bird->bird", "+sky->sky", "+fly->eagle"],
}).encode("utf-8")
req = urllib.request.Request(url, data=data,
    headers={"Content-Type": "text/plain"})
print(urllib.request.urlopen(req).read())
```

Dovresti vedere `{"ok":true}` e una nuova riga nel Sheet.
