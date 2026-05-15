/**
 * Meeting Scribe — Always-on-top translation overlay.
 *
 * Loads the reader.html view from the meeting-scribe server in a
 * frameless, always-on-top window. Stays visible above fullscreen
 * apps including PowerPoint presentations.
 *
 * Usage:
 *   cd overlay && npm install && npm start
 *   npm start -- --url https://192.168.1.100:8080/reader
 *   npm start -- --width 800 --height 300 --x 0 --y 0
 *
 * Keyboard shortcuts:
 *   Super+Shift+F  — Toggle visibility
 *   Super+Shift+Q  — Quit
 */

const { app, BrowserWindow, globalShortcut, screen } = require('electron');

// Parse CLI args
const args = process.argv.slice(2);
function getArg(name, defaultValue) {
  const idx = args.indexOf('--' + name);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : defaultValue;
}

const SERVER_URL = getArg('url', 'https://localhost:8080/reader');
const WIN_WIDTH = parseInt(getArg('width', '800'), 10);
const WIN_HEIGHT = parseInt(getArg('height', '250'), 10);
const WIN_X = getArg('x', null);
const WIN_Y = getArg('y', null);

let win = null;

app.whenReady().then(() => {
  const display = screen.getPrimaryDisplay();
  const { width: screenW } = display.workAreaSize;

  // Default position: top-center of screen
  const x = WIN_X !== null ? parseInt(WIN_X, 10) : Math.round((screenW - WIN_WIDTH) / 2);
  const y = WIN_Y !== null ? parseInt(WIN_Y, 10) : 0;

  win = new BrowserWindow({
    width: WIN_WIDTH,
    height: WIN_HEIGHT,
    x: x,
    y: y,
    alwaysOnTop: true,
    frame: false,
    transparent: false,
    backgroundColor: '#1a1a1e',
    skipTaskbar: true,
    resizable: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Accept self-signed certs from the meeting-scribe server
  win.webContents.session.setCertificateVerifyProc((request, callback) => {
    callback(0); // 0 = accept
  });

  win.loadURL(SERVER_URL);

  // Super+Shift+F: toggle visibility
  globalShortcut.register('Super+Shift+F', () => {
    if (win.isVisible()) {
      win.hide();
    } else {
      win.show();
      win.focus();
    }
  });

  // Super+Shift+Q: quit
  globalShortcut.register('Super+Shift+Q', () => {
    app.quit();
  });

  // Keep window on top even after focus changes
  win.on('blur', () => {
    win.setAlwaysOnTop(true);
  });

  console.log(`Meeting Scribe overlay started`);
  console.log(`  URL: ${SERVER_URL}`);
  console.log(`  Size: ${WIN_WIDTH}x${WIN_HEIGHT} at (${x}, ${y})`);
  console.log(`  Toggle: Super+Shift+F`);
  console.log(`  Quit: Super+Shift+Q`);
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

app.on('window-all-closed', () => {
  app.quit();
});
