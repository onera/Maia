This branch is only here to publish documentation on github.io server.

Follow these steps to do it:

- Copy the generated html directory of each version in `docs/` folder
- Create a file named `.nojekyll` in each version folder. Otherwise, javascript rendering will be
  desactivated
- Create of update the `docs/index.html` file to make it redirect to lastest doc version, for exemple :
  ```html
  <head>
    <meta http-equiv='refresh' content='0; URL=1.3/index.html'>
  </head>
  ```
- The link to others version is hardcoded in each html file. For github, it must be replaced by `/Maia/v_id`. To do that, execute this line in `docs/` directory: 
 `find . -name '*.html' -exec sed -i 's@href="/mesh/maia/@href="/Maia/@' {} +`
- To delete the reference to dev version, use in the same way
 `find . -name '*.html' -exec sed -i '/<dd><a href="\/Maia\/dev\/">/d' {} +`
