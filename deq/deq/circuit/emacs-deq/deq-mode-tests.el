;;; deq-mode-tests.el --- ERT tests for deq-mode -*- lexical-binding: t -*-

;; Run with:
;;   emacs -Q --batch -L . -l deq-mode-tests.el -f ert-run-tests-batch-and-exit
;;
;; Face assertions live in annotated files under `resources/'; see
;; `ert-font-lock'.

;;; Code:

(require 'ert)
(require 'ert-x)
(require 'ert-font-lock)
(require 'imenu)
(require 'deq-mode)

;;; ── Helpers ───────────────────────────────────────────────────────────

(defmacro deq-test-with-buffer (text &rest body)
  "Insert TEXT into a temp buffer in `deq-mode', fontify, run BODY."
  (declare (indent 1) (debug t))
  `(with-temp-buffer
     (deq-mode)
     (insert ,text)
     (font-lock-ensure)
     (goto-char (point-min))
     ,@body))

(defun deq-test--state-at (regexp)
  "Return the `syntax-ppss' state just after the first match of REGEXP."
  (goto-char (point-min))
  (re-search-forward regexp)
  (syntax-ppss))

(defun deq-test--reindent (text)
  "Insert TEXT in a `deq-mode' buffer, strip indentation, then re-indent it.
Return the resulting buffer contents."
  (with-temp-buffer
    (deq-mode)
    (insert text)
    (goto-char (point-min))
    (while (re-search-forward "^[ \t]+" nil t) (replace-match ""))
    (indent-region (point-min) (point-max))
    (buffer-string)))

;;; ── Activation ────────────────────────────────────────────────────────

(ert-deftest deq-mode/auto-mode-alist-triggers ()
  (with-temp-buffer
    (let ((buffer-file-name "/tmp/example.deq"))
      (set-auto-mode)
      (should (eq major-mode 'deq-mode)))))

(ert-deftest deq-mode/derives-from-prog-mode ()
  (with-temp-buffer
    (deq-mode)
    (should (derived-mode-p 'prog-mode))))

(ert-deftest deq-mode/sets-comment-syntax-vars ()
  (with-temp-buffer
    (deq-mode)
    (should (equal comment-start "# "))
    (should (string-match-p "#" comment-start-skip))))

;;; ── Font-lock ─────────────────────────────────────────────────────────

(ert-font-lock-deftest-file deq-mode/fontify-targets
  deq-mode "targets.deq")

(ert-font-lock-deftest-file deq-mode/fontify-statements
  deq-mode "statements.deq")

(ert-font-lock-deftest-file deq-mode/fontify-mako
  deq-mode "mako.deq")

(ert-deftest deq-mode/fontify-mako-without-python ()
  (let ((deq-mako-fontify-natively nil))
    (ert-font-lock-test-file (ert-resource-file "mako-plain.deq") 'deq-mode)))

(ert-deftest deq-mode/mako-python-partial-refontify-uses-block-context ()
  (with-temp-buffer
    (deq-mode)
    (insert "<%\ndoc = '''\nfor x\n'''\n%>\n")
    (goto-char (point-min))
    (re-search-forward "^for")
    (let ((for-start (match-beginning 0)))
      (font-lock-ensure (pos-bol) (pos-eol))
      (should (eq (get-text-property for-start 'face)
                  'font-lock-string-face)))))

;;; ── Syntax ────────────────────────────────────────────────────────────

(ert-deftest deq-mode/recognizes-comments-and-strings ()
  (deq-test-with-buffer "IMPORT \"abc.deq\"\nGADGET Foo {\n  # a comment\n}\n"
    (should (nth 3 (deq-test--state-at "abc")))
    (should (nth 4 (deq-test--state-at "comment")))))

(ert-deftest deq-mode/brackets-form-sexps ()
  (deq-test-with-buffer "CODE Foo[[2,1,3]] { X 0 }\n"
    (re-search-forward "{")
    (backward-char)
    (forward-sexp)
    (should (eq (char-before) ?\}))
    (goto-char (point-min))
    (re-search-forward "\\[\\[")
    (backward-char 2)
    (forward-sexp)
    (should (eq (char-before) ?\]))))

(ert-deftest deq-mode/tag-is-string ()
  (deq-test-with-buffer "GADGET F {\n  H[my # tag (x] 0\n  CX 0 1\n}\n"
    (should (nth 3 (deq-test--state-at "my")))
    (let ((state (deq-test--state-at "CX")))
      (should-not (nth 8 state))
      (should (= 1 (car state))))))

(ert-deftest deq-mode/record-and-code-params-are-not-tags ()
  (deq-test-with-buffer "CODE Foo[[2,1]] {}\nGADGET F {\n  DETECTOR rec[-1] sweep[0]\n}\n"
    (should-not (nth 8 (deq-test--state-at "2,")))
    (should-not (nth 8 (deq-test--state-at "rec\\[")))))

(ert-deftest deq-mode/mako-constructs-are-literals ()
  (deq-test-with-buffer
      (concat "<%\nx = {'#': '\"'}\n%>\n"
              "GADGET F {\n  % for i in range(n):  # {\n  M ${ {'}': 1}['}'] }\n"
              "  % endfor\n}\n")
    (should (eq (deq--mako-kind (progn (deq-test--state-at "x =") (point))) 'block))
    (should (eq (deq--mako-kind (progn (deq-test--state-at "for i") (point))) 'line))
    (should (eq (deq--mako-kind (progn (deq-test--state-at "\\${ {") (point))) 'expression))
    (let ((state (deq-test--state-at "^  M")))
      (should-not (nth 8 state))
      (should (= 1 (car state))))
    (should (= 0 (car (syntax-ppss (point-max)))))))

(ert-deftest deq-mode/unbalanced-mako-expression-stays-code ()
  (deq-test-with-buffer "GADGET F {\n  M ${foo(\n  CX 0 1\n}\n"
    (should-not (deq--mako-kind (progn (deq-test--state-at "foo") (point))))))

;;; ── Indentation and filling ───────────────────────────────────────────

(ert-deftest deq-mode/indent-nested-blocks ()
  (let ((expected (concat "GADGET Foo {\n"
                          "    READOUT R0\n"
                          "    REPEAT 3 {\n"
                          "        CX 0 1\n"
                          "    }\n"
                          "}\n")))
    (should (equal (deq-test--reindent expected) expected))))

(ert-deftest deq-mode/indent-respects-custom-offset ()
  (let ((deq-indent-offset 2)
        (expected "GADGET Foo {\n  REPEAT 3 {\n    CX 0 1\n  }\n}\n"))
    (should (equal (deq-test--reindent expected) expected))))

(ert-deftest deq-mode/indent-closing-brace-dedents ()
  (with-temp-buffer
    (deq-mode)
    (insert "GADGET Foo {\n        READOUT R0\n        }\n")
    (goto-char (point-min))
    (forward-line 2)
    (deq-indent-line)
    (should (= (current-indentation) 0))))

(ert-deftest deq-mode/indent-preserves-mako-python ()
  (let ((text (concat "<%\n"
                      "if d > 3:\n"
                      "    r = {\n"
                      "        'a': 1,\n"
                      "    }\n"
                      "%>\n"
                      "GADGET Foo {\n"
                      "% for i in range(d):\n"
                      "    M ${i}\n"
                      "        % if i:\n"
                      "    CX 0 1\n"
                      "        % endif\n"
                      "% endfor\n"
                      "}\n")))
    (with-temp-buffer
      (deq-mode)
      (insert text)
      (indent-region (point-min) (point-max))
      (should (equal (buffer-string) text)))))

(ert-deftest deq-mode/fill-paragraph-leaves-mako-python ()
  (let ((text "<%\nx = 1\ny = 2\n%>\n"))
    (with-temp-buffer
      (deq-mode)
      (insert text)
      (goto-char (point-min))
      (re-search-forward "x =")
      (fill-paragraph)
      (should (equal (buffer-string) text)))))

;;; ── Navigation ────────────────────────────────────────────────────────

(ert-deftest deq-mode/imenu-finds-all-definition-kinds ()
  (deq-test-with-buffer
      (concat "CODE C[[2,1]] {\n  LOGICAL X0 Z0\n}\n"
              "GADGET G {\n}\n"
              "COMPOSE M {\n}\n"
              "PROGRAM P {\n}\n"
              "<%def name=\"rounds(n)\">\n</%def>\n"
              "GADGET Round${d} {\n}\n")
    (let ((index (imenu--make-index-alist)))
      (should (assoc "C" (cdr (assoc "Codes" index))))
      (should (assoc "G" (cdr (assoc "Gadgets" index))))
      (should (assoc "Round${d}" (cdr (assoc "Gadgets" index))))
      (should (assoc "M" (cdr (assoc "Compose" index))))
      (should (assoc "P" (cdr (assoc "Programs" index))))
      (should (assoc "rounds" (cdr (assoc "Mako defs" index)))))))

(ert-deftest deq-mode/defun-navigation ()
  (deq-test-with-buffer
      (concat "GADGET A {\n  REPEAT 2 {\n    CX 0 1\n  }\n}\n"
              "# GADGET NotADefinition {\n"
              "@GTYPE(2)\n@Other\nGADGET B {\n  X 0\n}\n")
    (re-search-forward "^  X 0")
    (beginning-of-defun)
    (should (looking-at-p "@GTYPE"))
    (beginning-of-defun)
    (should (looking-at-p "GADGET A"))
    (end-of-defun)
    (should (looking-at-p "# GADGET NotADefinition"))
    (goto-char (point-min))
    (beginning-of-defun -1)
    (should (looking-at-p "@GTYPE"))))

;;; ── Comment commands ─────────────────────────────────────────────────

(ert-deftest deq-mode/comment-region-uses-hash ()
  (with-temp-buffer
    (deq-mode)
    (insert "CX 0 1\n")
    (comment-region (point-min) (point-max))
    (goto-char (point-min))
    (should (looking-at "#"))))

(provide 'deq-mode-tests)

;;; deq-mode-tests.el ends here
