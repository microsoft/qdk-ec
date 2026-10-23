;;; deq-mode.el --- Major mode for DEQ quantum error correction files -*- lexical-binding: t -*-

;; Copyright (C) 2026 Microsoft Corporation

;; Author: Juan M. Bello-Rivas <jbellorivas@microsoft.com>
;; URL: https://github.com/microsoft/qdk-ec
;; Keywords: languages
;; Package-Requires: ((emacs "30.1"))
;; Version: 0.2.0
;; SPDX-License-Identifier: MIT

;; This file is not part of GNU Emacs.

;;; Commentary:

;; Major mode for editing `.deq' files — the domain specific language for the
;; DEQ quantum error correction decoding system.  Features:
;;
;;   * Syntax highlighting for all DEQ constructs (CODE, GADGET,
;;     COMPOSE, PROGRAM blocks; LOGICAL/STABILIZER declarations;
;;     INPUT/OUTPUT ports; READOUT/CHECK/ERROR/LOSS/CONDITIONAL
;;     statements; embedded Stim instructions, tags and `rad'
;;     arguments; decorators and their keyword arguments).
;;   * Distinct faces for each target class (Pauli, check, readout,
;;     logical, measurement record, loss) mirroring the VS Code
;;     extension's color scheme.
;;   * Warnings for PRESELECT forms the parser rejects.
;;   * Mako template support.  `<% ... %>' blocks and `%' control lines
;;     are syntactic comments and `${...}' expressions are strings, so
;;     their Python never affects DEQ parsing or indentation.  The Python
;;     inside them is highlighted by `python-mode', like Org source
;;     blocks (see `deq-mako-fontify-natively').
;;   * `{'-aware indentation built on `syntax-ppss'.
;;   * Imenu navigation for CODE / GADGET / COMPOSE / PROGRAM names
;;     and Mako `<%def>' blocks.
;;   * Auto-registers for files ending in `.deq'.
;;
;; Install:
;;
;;   (use-package deq-mode
;;     :load-path "/path/to/emacs-deq"
;;     :mode ("\\.deq\\'" . deq-mode))

;;; Code:

(defgroup deq nil
  "Major mode for DEQ quantum error correction files."
  :group 'languages
  :prefix "deq-")

(defcustom deq-indent-offset 4
  "Number of spaces per indentation level in `deq-mode'."
  :type 'integer
  :safe #'integerp)

(defcustom deq-mako-fontify-natively t
  "Non-nil means highlight Python inside Mako constructs natively.
The Python in `<% ... %>' blocks, `${...}' expressions and `%' control
lines is highlighted by `deq-mako-python-mode', like Org source blocks."
  :type 'boolean
  :safe #'booleanp)

(defcustom deq-mako-python-mode #'python-mode
  "Major mode used to highlight Python inside Mako constructs."
  :type 'function)

;;; ── Faces ─────────────────────────────────────────────────────────────

(defface deq-pauli-face
  '((t :inherit font-lock-type-face))
  "Face for Pauli operators (e.g. `X0', `Y3', `Z7', `I2').")

(defface deq-logical-face
  '((t :inherit font-lock-function-name-face))
  "Face for logical targets (e.g. `LX0', `IN0.LZ1', `OUT1.L4').")

(defface deq-check-face
  '((t :inherit font-lock-constant-face))
  "Face for checks and stabilizer references (e.g. `C0', `IN0.S2', `IN0.DS2').")

(defface deq-readout-face
  '((t :inherit font-lock-variable-name-face))
  "Face for readout targets (e.g. `R0').")

(defface deq-record-face
  '((t :inherit font-lock-variable-name-face))
  "Face for measurement targets (e.g. `rec[-1]', `M3', `sweep[2]').")

(defface deq-loss-face
  '((t :inherit font-lock-constant-face))
  "Face for loss targets (e.g. `SE0', `CE1', `L2').")

(defface deq-decorator-face
  '((t :inherit font-lock-preprocessor-face))
  "Face for decorators (e.g. `@GTYPE(2)').")

(defface deq-mako-face
  '((t :inherit font-lock-preprocessor-face))
  "Face for Mako template delimiters (e.g. `${', `<%', `%>', `% for').")

;;; ── Regexp building blocks ────────────────────────────────────────────

(rx-define deq--identifier
  (seq (in "A-Za-z") (* (in "A-Za-z0-9_"))))

(rx-define deq--name
  (+ (or deq--identifier (seq "${" (* (not (in "}\n"))) "}"))))

(rx-define deq--indexed (prefix)
  (seq symbol-start prefix (+ digit) symbol-end))

(rx-define deq--port-target (ports kind)
  (seq symbol-start ports (+ digit) "." kind (+ digit) symbol-end))

(rx-define deq--number
  (seq (? "-")
       (or (seq (+ digit) (? "." (* digit))) (seq "." (+ digit)))
       (? (in "eE") (? (in "+-")) (+ digit))))

(rx-define deq--definition-keyword
  (or "CODE" "GADGET" "COMPOSE" "PROGRAM"))

(defconst deq--control-keywords
  '("IMPORT" "REPEAT")
  "Control-flow / file-level keywords.")

(defconst deq--statement-keywords
  '("INPUT" "OUTPUT"
    "LOGICAL" "STABILIZER"
    "READOUT" "OBSERVABLE_INCLUDE"
    "CHECK" "DETECTOR"
    "ERROR" "LOSS"
    "CONDITIONAL" "PRESELECT"
    "VIRTUAL" "PROPAGATE" "FROM" "FLIP"
    "ASSERT_EQ"
    "IN" "OUT")
  "Body-statement keywords used inside definitions.")

;;; ── Syntax ────────────────────────────────────────────────────────────

(defvar deq-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?#  "<"  table)
    (modify-syntax-entry ?\n ">"  table)
    (modify-syntax-entry ?_  "w"  table)
    (modify-syntax-entry ?\" "\"" table)
    ;; DEQ never uses `''; Python inside `${...}' does.
    (modify-syntax-entry ?\' "\"" table)
    (modify-syntax-entry ?\( "()" table)
    (modify-syntax-entry ?\) ")(" table)
    (modify-syntax-entry ?\[ "(]" table)
    (modify-syntax-entry ?\] ")[" table)
    (modify-syntax-entry ?\{ "(}" table)
    (modify-syntax-entry ?\} "){" table)
    (modify-syntax-entry ?!  "."  table)
    (modify-syntax-entry ?@  "."  table)
    (modify-syntax-entry ?*  "."  table)
    table)
  "Syntax table for `deq-mode'.")

(defun deq--literal-start-at-match ()
  "Return the start of the string or comment containing the current match."
  (save-excursion
    (save-match-data
      (nth 8 (syntax-ppss (match-beginning 0))))))

(defun deq--fence-unless-literal (syntax)
  "Return SYNTAX as a syntax descriptor unless the match is in a literal."
  (unless (deq--literal-start-at-match)
    (string-to-syntax syntax)))

(defun deq--mako-kind (&optional pos)
  "Return the kind of Mako construct containing POS (default point).
The value is `block', `line', `expression' or nil."
  (when-let* ((start (nth 8 (save-excursion (syntax-ppss pos)))))
    (pcase (char-after start)
      (?< 'block)
      (?% 'line)
      (?$ 'expression))))

(defun deq--mako-block-closer-syntax ()
  "Return the syntax closing a Mako block at the current `%>' match."
  (when-let* ((start (deq--literal-start-at-match))
              ((eq (char-after start) ?<)))
    (string-to-syntax "!")))

(defun deq--mako-expression-syntax ()
  "Return the syntax opening a `${...}' at the current match.
Also fence its closing `}', which must be on the same line."
  (unless (deq--literal-start-at-match)
    (when-let* ((close (save-match-data
                         (ignore-errors
                           (scan-lists (1+ (match-beginning 0)) 1 0))))
                ((<= close (pos-eol))))
      (put-text-property (1- close) close 'syntax-table (string-to-syntax "|"))
      (string-to-syntax "|"))))

(defun deq--tag-syntax ()
  "Return the syntax of a Stim instruction tag bracket at the current match."
  (unless (or (string-match-p (rx bos (or "rec" "sweep") "[") (match-string 0))
              (deq--literal-start-at-match))
    (string-to-syntax "|")))

(defconst deq-syntax-propertize-function
  (syntax-propertize-rules
   ((rx (group "<") "%" (or (not (in "A-Za-z/")) eol))
    (1 (deq--fence-unless-literal "!")))
   ((rx "%" (group ">"))
    (1 (deq--mako-block-closer-syntax)))
   ((rx bol (* blank) (group "%") (or (not (in "%>\n")) eol) (* nonl) (group "\n"))
    (1 (deq--fence-unless-literal "!"))
    (2 (deq--fence-unless-literal "!")))
   ((rx (group "$") "{")
    (1 (deq--mako-expression-syntax)))
   ((rx symbol-start deq--identifier
        (group "[")
        (? (or (not (in "[]$\n")) (seq "$" (not (in "{]\n"))))
           (* (or (not (in "]$\n")) (seq "$" (not (in "{]\n"))))))
        (group "]"))
    (1 (deq--tag-syntax))
    (2 (deq--tag-syntax))))
  "Mark Mako constructs and Stim tags.
`<% ... %>' blocks and `%' control lines become comments, `${...}'
expressions and tags become strings, so their contents never start
DEQ comments, strings or lists.  Tags cannot contain `${', which
always starts Mako, as in `rec[${-1-i}]'.")

(defun deq--syntactic-face (state)
  "Return the syntactic face for STATE, leaving Mako constructs unstyled."
  (cond ((memq (char-after (nth 8 state)) '(?< ?% ?$)) nil)
        ((nth 3 state) 'font-lock-string-face)
        (t 'font-lock-comment-face)))

;;; ── Native Python highlighting in Mako ────────────────────────────────

(defvar python-indent-guess-indent-offset)

(defun deq--mako-python-buffer ()
  "Return the hidden buffer used to highlight Mako Python code."
  (let ((buffer (get-buffer-create " *deq-mako-python*")))
    (with-current-buffer buffer
      (unless (eq major-mode deq-mako-python-mode)
        (let ((python-indent-guess-indent-offset nil))
          (delay-mode-hooks (funcall deq-mako-python-mode)))))
    buffer))

(defun deq--fontify-python-region (start end)
  "Highlight the text between START and END as Python.
The text is highlighted in a hidden buffer whose faces are copied back,
replacing any DEQ faces."
  (when (< start end)
    (let ((fontified (let ((code (buffer-substring-no-properties start end)))
                       (with-current-buffer (deq--mako-python-buffer)
                         (erase-buffer)
                         (insert code)
                         (font-lock-ensure)
                         (buffer-string)))))
      (remove-text-properties start end '(face nil))
      (pcase-dolist (`(,from ,to ,properties) (object-intervals fontified))
        (when-let* ((face (plist-get properties 'face)))
          (put-text-property (+ start from) (+ start to) 'face face))))))

(defun deq--mako-construct-end (open)
  "Return the end of the Mako construct opened at OPEN, or nil if unclosed."
  (save-excursion
    (let ((state (syntax-ppss (1+ open))))
      (parse-partial-sexp (1+ open) (point-max) nil nil state 'syntax-table)
      (unless (nth 8 (syntax-ppss))
        (point)))))

(defun deq--fontify-mako-construct (open kind end)
  "Highlight the Mako construct of KIND spanning OPEN to END."
  (let ((code-start (+ open (if (eq kind 'line) 1 2)))
        (code-end (- end (pcase kind ('block 2) (_ 1)))))
    (when (and (eq kind 'block) (eq (char-after code-start) ?!))
      (setq code-start (1+ code-start)))
    (put-text-property open code-start 'face 'deq-mako-face)
    (unless (eq kind 'line)
      (put-text-property code-end end 'face 'deq-mako-face))
    (setq code-start (save-excursion
                       (goto-char code-start)
                       (skip-chars-forward " \t" code-end)
                       (point)))
    (if (and (eq kind 'line)
             (or (not deq-mako-fontify-natively)
                 (save-excursion
                   (goto-char code-start)
                   (looking-at-p (rx "end" (* (in "a-z")) symbol-end)))))
        (save-excursion
          (goto-char code-start)
          (when (looking-at (rx (+ (in "a-z")) symbol-end))
            (put-text-property (match-beginning 0) (match-end 0)
                               'face 'font-lock-keyword-face)))
      (when deq-mako-fontify-natively
        (deq--fontify-python-region code-start code-end)))))

(defun deq--fontify-mako (limit)
  "Font-lock matcher highlighting Mako constructs from point to LIMIT.
Always returns nil: the faces are applied directly."
  (while (re-search-forward (rx (in "<%$")) limit t)
    (let ((open (1- (point))))
      (when-let* (((eq (nth 8 (save-excursion (syntax-ppss (point)))) open))
                  (kind (deq--mako-kind (point)))
                  (end (deq--mako-construct-end open)))
        (deq--fontify-mako-construct open kind end)
        (goto-char end))))
  nil)

(defvar font-lock-beg)
(defvar font-lock-end)

(defun deq--extend-region-to-mako-blocks ()
  "Extend the font-lock region to whole Mako `<% ... %>' blocks.
Python must be highlighted with its full block as context, such as
the opening line of a multi-line string."
  (let ((beg font-lock-beg)
        (end font-lock-end))
    (when (eq (deq--mako-kind beg) 'block)
      (setq font-lock-beg (save-excursion
                            (goto-char (nth 8 (syntax-ppss beg)))
                            (pos-bol))))
    (when-let* (((eq (deq--mako-kind end) 'block))
                (block-end (deq--mako-construct-end
                            (nth 8 (save-excursion (syntax-ppss end))))))
      (setq font-lock-end (max end block-end)))
    (or (/= beg font-lock-beg) (/= end font-lock-end))))

;;; ── Font-lock ─────────────────────────────────────────────────────────

(defun deq--search-outside-comments (regexp limit)
  "Search forward for REGEXP before LIMIT, skipping matches in comments."
  (let ((found nil))
    (while (and (not found) (re-search-forward regexp limit t))
      (setq found (not (save-match-data
                         (nth 4 (save-excursion
                                  (syntax-ppss (match-beginning 0))))))))
    found))

(defun deq--match-preselect-virtual-stabilizer (limit)
  "Match a virtual stabilizer target, which PRESELECT rejects, before LIMIT."
  (deq--search-outside-comments
   (rx (deq--port-target (or "IN" "OUT") "S")) limit))

(defun deq--match-preselect-invalid-parity (limit)
  "Match a PRESELECT parity other than 0 or 1 before LIMIT."
  (deq--search-outside-comments
   (rx symbol-start (or (in "2-9") (seq (in "1-9") (+ digit))) symbol-end
       (* blank) (? "#" (* nonl)) eol)
   limit))

(defconst deq-font-lock-keywords
  `((,(rx (group "<%") (group "def") symbol-end)
     (1 'deq-mako-face) (2 'font-lock-keyword-face))
    (,(rx (group "</%") (group "def") (group ">"))
     (1 'deq-mako-face) (2 'font-lock-keyword-face) (3 'deq-mako-face))

    (,(rx "@" deq--identifier)
     (0 'deq-decorator-face)
     (,(rx symbol-start (group deq--identifier) (* blank) "=") nil nil
      (1 'font-lock-variable-name-face)))

    (,(rx word-boundary (group deq--definition-keyword) (+ space) (group deq--name))
     (1 'font-lock-keyword-face)
     (2 'font-lock-function-name-face keep))
    (,(regexp-opt deq--control-keywords 'symbols) 0 'font-lock-keyword-face)
    (,(regexp-opt deq--statement-keywords 'symbols) 0 'font-lock-builtin-face)

    ("!" 0 'font-lock-negation-char-face)

    ;; More specific targets first: the first match wins.
    (,(rx (deq--port-target (or "IN" "OUT") (seq "L" (? (in "XYZ")))))
     0 'deq-logical-face)
    (,(rx (deq--port-target "IN" "DS")) 0 'deq-check-face)
    (,(rx (deq--port-target (or "IN" "OUT") "S")) 0 'deq-check-face)
    (,(rx (deq--indexed (seq "L" (in "XYZ")))) 0 'deq-logical-face)
    (,(rx (deq--indexed (or "SE" "CE" "L"))) 0 'deq-loss-face)
    (,(rx (deq--indexed "M")) 0 'deq-record-face)
    (,(rx (deq--indexed "C")) 0 'deq-check-face)
    (,(rx (deq--indexed "R")) 0 'deq-readout-face)
    (,(rx (deq--indexed (in "IXYZ"))) 0 'deq-pauli-face)
    (,(rx (or (seq "rec[-" (+ digit) "]") (seq "sweep[" (+ digit) "]")))
     0 'deq-record-face)
    (,(rx symbol-start deq--number (? "rad") symbol-end) 0 'font-lock-constant-face)

    ;; PRESELECT rejects virtual stabilizer references and parities
    ;; other than 0 and 1.
    (,(rx symbol-start "PRESELECT" symbol-end)
     (deq--match-preselect-virtual-stabilizer nil nil
      (0 'font-lock-warning-face t))
     (deq--match-preselect-invalid-parity
      (prog1 (pos-eol) (goto-char (match-end 0))) nil
      (0 'font-lock-warning-face t)))

    ;; Runs last to replace DEQ faces inside Mako constructs.
    (deq--fontify-mako))
  "Font-lock keywords for `deq-mode'.")

;;; ── Indentation and filling ───────────────────────────────────────────

(defun deq--mako-python-line-p ()
  "Non-nil if the current line is Mako Python, whose indentation is kept.
That is a line inside or opening a `<% ... %>' block, or a `%' line."
  (save-excursion
    (back-to-indentation)
    (or (eq (deq--mako-kind (pos-bol)) 'block)
        (and (not (eolp))
             (memq (deq--mako-kind (1+ (point))) '(block line))))))

(defun deq-indent-line ()
  "Indent the current line of DEQ source.

Indent to the `{'/`('/`[' nesting depth times `deq-indent-offset',
dedenting lines that start with a closer by one level.  Mako Python
lines keep their indentation."
  (interactive)
  (if (deq--mako-python-line-p)
      'noindent
    (let* ((depth (car (syntax-ppss (pos-bol))))
           (closer (save-excursion
                     (back-to-indentation)
                     (memq (char-after) '(?\} ?\) ?\]))))
           (target (* (max 0 (if closer (1- depth) depth)) deq-indent-offset)))
      (if (<= (current-column) (current-indentation))
          (indent-line-to target)
        (save-excursion (indent-line-to target))))))

(defun deq--fill-paragraph (&optional _justify)
  "Leave Mako Python unfilled: it is code, not a comment or string."
  (and (deq--mako-kind) t))

;;; ── Navigation ────────────────────────────────────────────────────────

(defconst deq-imenu-generic-expression
  `(("Codes"     ,(rx bol (* blank) "CODE"    (+ blank) (group deq--name)) 1)
    ("Gadgets"   ,(rx bol (* blank) "GADGET"  (+ blank) (group deq--name)) 1)
    ("Compose"   ,(rx bol (* blank) "COMPOSE" (+ blank) (group deq--name)) 1)
    ("Programs"  ,(rx bol (* blank) "PROGRAM" (+ blank) (group deq--name)) 1)
    ("Mako defs" ,(rx bol (* blank) "<%def" (+ blank) "name=\""
                      (group (+ (not (in "\"(")))))
     1))
  "Imenu patterns for navigating DEQ definitions.")

(defun deq--search-definition (backward)
  "Move to the previous (if BACKWARD) or next top-level definition.
Point ends at its first decorator, if any.  Return non-nil if found."
  (let ((keyword-regexp (rx bol (* blank) deq--definition-keyword symbol-end))
        (found nil))
    (unless backward
      (while (looking-at-p (rx (* blank) "@"))
        (forward-line 1))
      (end-of-line))
    (while (and (not found)
                (if backward
                    (re-search-backward keyword-regexp nil 'move)
                  (re-search-forward keyword-regexp nil 'move)))
      (let ((state (save-excursion (syntax-ppss (match-beginning 0)))))
        (setq found (and (zerop (car state)) (not (nth 8 state))))))
    (when found
      (goto-char (match-beginning 0))
      (while (and (not (bobp))
                  (save-excursion
                    (forward-line -1)
                    (looking-at-p (rx (* blank) "@"))))
        (forward-line -1)))
    found))

(defun deq-beginning-of-defun (&optional arg)
  "Move backward to the beginning of the enclosing DEQ definition.
With ARG, repeat that many times (or forward if ARG is negative)."
  (let ((arg (or arg 1))
        (found t))
    (dotimes (_ (abs arg))
      (setq found (deq--search-definition (> arg 0))))
    found))

(defun deq-end-of-defun ()
  "Move to the end of the current DEQ definition.
Assumes point is on or before the opening `{' of the definition."
  (when (re-search-forward "{" nil 'move)
    (backward-char)
    (ignore-error scan-error (forward-sexp))))

;;; ── Mode definition ───────────────────────────────────────────────────

;;;###autoload
(define-derived-mode deq-mode prog-mode "DEQ"
  "Major mode for editing DEQ quantum error correction source files.

\\{deq-mode-map}"
  (setq-local comment-start "# ")
  (setq-local comment-start-skip "#+\\s-*")
  (setq-local comment-end "")
  (setq-local comment-use-syntax t)
  (setq-local syntax-propertize-function deq-syntax-propertize-function)
  (setq-local font-lock-defaults
              '(deq-font-lock-keywords
                nil nil nil nil
                (font-lock-syntactic-face-function . deq--syntactic-face)))
  (add-hook 'font-lock-extend-region-functions
            #'deq--extend-region-to-mako-blocks nil t)
  (setq-local jit-lock-contextually t)
  (setq-local fill-paragraph-function #'deq--fill-paragraph)
  (setq-local indent-line-function #'deq-indent-line)
  (setq-local indent-tabs-mode nil)
  (setq-local imenu-generic-expression deq-imenu-generic-expression)
  (setq-local beginning-of-defun-function #'deq-beginning-of-defun)
  (setq-local end-of-defun-function #'deq-end-of-defun))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.deq\\'" . deq-mode))

(provide 'deq-mode)

;;; deq-mode.el ends here
